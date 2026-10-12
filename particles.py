"""
particles.py -- Phase 3: fill components with emitting particles.

SHAPE represents emission as a cloud of particles. Each particle carries:
  position (arcsec), emissivity weight, component ID, and velocity (km/s).

We sample particles ANALYTICALLY so they reproduce the Phase-1 grid shapes
exactly (this is what makes the particle<->grid cross-check meaningful):
  * shell : uniform spatial density in the ellipsoidal wall (same inner/outer
            ellipsoids as grid_model, defined by semi-axes +/- thickness/2),
  * lobes : uniform spatial density on the two hollow cone walls
            (axial pdf proportional to t so density per unit area is uniform),
  * knots : 3-D Gaussian blobs (truncated), so number density follows the
            Gaussian emissivity.

Velocities are homologous, v = H*r, assigned in the body frame; positions AND
velocity vectors are then rotated into the sky frame, so v_los = v_z.

By default NO modifiers are applied (so the particle model matches the grid).
Optional modifier stacks per component are supported for SHAPE-style shaping.
"""

from dataclasses import dataclass

import numpy as np

import config as cfg
from config import ID_SHELL, ID_LOBE, ID_KNOT
from orientation import body_to_sky_matrix
from velocity import hubble_constant
from grid_model import build_model


@dataclass
class ParticleSet:
    """A cloud of emitting particles (in whatever frame positions are given)."""
    positions: np.ndarray    # (N, 3) arcsec
    weights: np.ndarray      # (N,) emissivity weight
    comp_id: np.ndarray      # (N,) integer component ID
    velocities: np.ndarray   # (N, 3) km/s

    def __len__(self):
        return len(self.weights)

    @property
    def v_los(self):
        """Line-of-sight velocity = z-component of velocity (km/s)."""
        return self.velocities[:, 2]

    def select(self, mask):
        return ParticleSet(self.positions[mask], self.weights[mask],
                           self.comp_id[mask], self.velocities[mask])


# --- Analytic volume samplers (BODY frame) --------------------------------

def sample_sphere_directions(n, rng):
    """n uniformly distributed unit vectors on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample_shell(shell, n, rng):
    """Uniform-density points in the ellipsoidal wall (same as grid_model shell).

    Strategy: rejection sampling inside the bounding box of the outer ellipsoid,
    accepting points that lie inside the outer ellipsoid and outside the inner
    one -- identical condition to grid_model.shell_emissivity.
    """
    ax, ay, az = shell.semi_axes_arcsec
    t = shell.thickness_arcsec
    ao = np.array([ax + t / 2, ay + t / 2, az + t / 2])   # outer semi-axes
    ai = np.array([ax - t / 2, ay - t / 2, az - t / 2])   # inner semi-axes
    pts = np.empty((0, 3))
    while len(pts) < n:
        batch = rng.uniform(-ao, ao, size=(n * 2, 3))
        q_out = np.sum((batch / ao) ** 2, axis=1)
        q_in = np.sum((batch / ai) ** 2, axis=1)
        keep = (q_out <= 1.0) & (q_in >= 1.0)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def sample_lobes(lobe, n, rng):
    """Uniform-density points on the two hollow cone walls (same as grid_model).

    Axial coordinate t is drawn with pdf proportional to t (so that density per
    unit wall area is uniform, since the wall circumference grows like t). The
    radius scatters within +/- wall_thickness/2 of the cone surface t*tan(theta).
    Half the particles go to each lobe (+y and -y).
    """
    theta = np.deg2rad(lobe.half_opening_angle_deg)
    t0, t1 = lobe.inner_cutoff_arcsec, lobe.length_arcsec
    u = rng.uniform(0, 1, n)
    t = np.sqrt(t0 ** 2 + u * (t1 ** 2 - t0 ** 2))         # pdf(t) ~ t
    phi = rng.uniform(0, 2 * np.pi, n)
    rho = t * np.tan(theta) + rng.uniform(-lobe.wall_thickness_arcsec / 2,
                                          lobe.wall_thickness_arcsec / 2, n)
    sign = rng.choice([1.0, -1.0], n)
    x = rho * np.cos(phi)
    y = sign * t
    z = rho * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def sample_knots(knot, n, rng):
    """Truncated 3-D Gaussian blobs at the lobe tips (0, +/- p, 0)."""
    p = knot.axial_position_arcsec
    sig = knot.sigma_arcsec
    rtrunc = knot.truncate_nsigma * sig
    out = np.empty((0, 3))
    while len(out) < n:
        batch = rng.normal(0, sig, size=(n * 2, 3))
        keep = np.linalg.norm(batch, axis=1) <= rtrunc
        out = np.vstack([out, batch[keep]])
    out = out[:n]
    sign = rng.choice([1.0, -1.0], len(out))
    out[:, 1] += sign * p                                   # shift to +/- tip
    return out


# --- Build the particle model ---------------------------------------------

def build_particle_model(config=None, inclination_deg=None, position_angle_deg=None,
                         n_per_component=None, modifiers=None, match_grid_flux=True):
    """Build the particle model in the SKY frame.

    modifiers: optional dict {component_id: ModifierStack} applied in the body
               frame before rotation (default: none, so it matches the grid).
    match_grid_flux: if True, scale each component's particle weights so the
               component's total emission equals the Phase-1 grid total (makes
               combined images directly comparable to the grid).

    Returns a dict with:
      particles   : combined ParticleSet (sky frame)
      components  : {id: ParticleSet (sky frame)}
      H           : {id: homologous constant}
      inclination_deg, position_angle_deg, config
    """
    if config is None:
        config = cfg.DEFAULT
    if inclination_deg is None:
        inclination_deg = config.geometry.inclination_deg
    if position_angle_deg is None:
        position_angle_deg = config.geometry.position_angle_deg
    if n_per_component is None:
        n_per_component = config.n_particles_per_component
    modifiers = modifiers or {}

    rng = np.random.default_rng(config.seed)

    # Homologous constants (same as grid_model).
    H = {
        ID_SHELL: hubble_constant(config.shell.expansion_velocity_kms,
                                  config.shell.v_char_radius_arcsec),
        ID_LOBE: hubble_constant(config.lobe.tip_velocity_kms,
                                 config.lobe.v_char_radius_arcsec),
        ID_KNOT: hubble_constant(config.knot.velocity_kms,
                                 config.knot.v_char_radius_arcsec),
    }
    base_emiss = {ID_SHELL: config.shell.emissivity,
                  ID_LOBE: config.lobe.emissivity,
                  ID_KNOT: config.knot.emissivity}

    samplers = {
        ID_SHELL: lambda n: sample_shell(config.shell, n, rng),
        ID_LOBE: lambda n: sample_lobes(config.lobe, n, rng),
        ID_KNOT: lambda n: sample_knots(config.knot, n, rng),
    }

    # Optionally match per-component total flux to the grid model.
    grid_totals = None
    if match_grid_flux:
        grid = build_model(config, inclination_deg, position_angle_deg)
        grid_totals = {cid: float(c["emissivity"].sum())
                       for cid, c in grid["components"].items()}

    R = body_to_sky_matrix(inclination_deg, position_angle_deg)
    components = {}
    for cid, sampler in samplers.items():
        pts_body = sampler(n_per_component)                 # (N,3) body frame
        if cid in modifiers and not modifiers[cid].is_empty():
            pts_body = modifiers[cid](pts_body)

        # Homologous velocity in the body frame: v = H * r.
        vel_body = H[cid] * pts_body

        # Rotate positions and velocities into the sky frame.
        pts_sky = pts_body @ R.T
        vel_sky = vel_body @ R.T

        # Per-particle weight; optionally rescale so sum matches grid total.
        w = np.full(len(pts_sky), base_emiss[cid], dtype=float)
        if grid_totals is not None and w.sum() > 0:
            w *= grid_totals[cid] / w.sum()

        comp_id = np.full(len(pts_sky), cid, dtype=int)
        components[cid] = ParticleSet(pts_sky, w, comp_id, vel_sky)

    # Combined set.
    combined = ParticleSet(
        np.vstack([c.positions for c in components.values()]),
        np.concatenate([c.weights for c in components.values()]),
        np.concatenate([c.comp_id for c in components.values()]),
        np.vstack([c.velocities for c in components.values()]),
    )

    return {"particles": combined, "components": components, "H": H,
            "inclination_deg": inclination_deg,
            "position_angle_deg": position_angle_deg, "config": config}
