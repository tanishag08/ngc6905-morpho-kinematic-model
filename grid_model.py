"""
grid_model.py -- Phase 1: simple analytic 3D emissivity grid model.

We build a cube of emissivity on a regular grid in SKY coordinates
(x=east, y=north, z=line of sight), all in arcsec. Each component is an
analytic function defined in the BODY frame (symmetry axis = body +y). For
every sky voxel we rotate its coordinates into the body frame (r_body = R^T r_sky)
and evaluate the analytic components there. This bakes the inclination / PA
orientation into the sampling while keeping the component shapes simple.

Components (each tagged with an integer ID from config):
  a) central shell -- ellipsoidal shell between an inner and outer ellipsoid
  b) bipolar lobes -- two opposing hollow cones along body +/- y
  c) knots         -- two Gaussian blobs at the lobe tips

The velocity field is homologous (see velocity.py): each component has its own
H = v_char / r_char, and the line-of-sight velocity of a voxel is H * z_sky.
"""

import numpy as np

import config as cfg
from orientation import sky_to_body
from velocity import hubble_constant


# --- Grid construction -----------------------------------------------------

def make_grid(grid_params):
    """Return sky-frame coordinate cubes (X, Y, Z) in arcsec and the 1-D axis.

    X, Y, Z each have shape (n, n, n). Indexing is 'ij' so X varies along axis 0,
    Y along axis 1, Z along axis 2.
    """
    n = grid_params.n
    half = grid_params.half_size_arcsec
    axis = np.linspace(-half, half, n)            # arcsec, same for x, y, z
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    return X, Y, Z, axis


def sky_cubes_to_body(X, Y, Z, inclination_deg, position_angle_deg):
    """Rotate every sky voxel (X,Y,Z) into body coordinates (Xb, Yb, Zb)."""
    pts_sky = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)
    pts_body = sky_to_body(pts_sky, inclination_deg, position_angle_deg)
    shape = X.shape
    Xb = pts_body[:, 0].reshape(shape)
    Yb = pts_body[:, 1].reshape(shape)
    Zb = pts_body[:, 2].reshape(shape)
    return Xb, Yb, Zb


# --- Analytic component emissivities (all in BODY frame, arcsec) ----------

def shell_emissivity(Xb, Yb, Zb, shell):
    """Ellipsoidal shell between an inner and outer ellipsoid.

    A point is in the shell if it lies inside the outer ellipsoid (semi-axes
    a_k + t/2) and outside the inner ellipsoid (semi-axes a_k - t/2), where t is
    the wall thickness. Emissivity is uniform inside the wall.
    """
    ax, ay, az = shell.semi_axes_arcsec
    t = shell.thickness_arcsec
    # Outer and inner ellipsoid "normalized radii" (=1 on the surface).
    q_out = np.sqrt((Xb / (ax + t / 2)) ** 2 +
                    (Yb / (ay + t / 2)) ** 2 +
                    (Zb / (az + t / 2)) ** 2)
    q_in = np.sqrt((Xb / (ax - t / 2)) ** 2 +
                   (Yb / (ay - t / 2)) ** 2 +
                   (Zb / (az - t / 2)) ** 2)
    in_wall = (q_out <= 1.0) & (q_in >= 1.0)
    emiss = np.where(in_wall, shell.emissivity, 0.0)
    return emiss


def lobe_emissivity(Xb, Yb, Zb, lobe):
    """Two opposing hollow cones along body +/- y (the symmetry axis).

    For a point, the axial coordinate is Yb and the cylindrical radius from the
    axis is rho = sqrt(Xb^2 + Zb^2). A hollow cone of half-opening angle theta
    has its wall at rho = |Yb| * tan(theta); we keep points within +/- half the
    wall thickness of that surface, between the inner cutoff and the lobe length.
    Using |Yb| makes both lobes (north and south) appear automatically.
    """
    theta = np.deg2rad(lobe.half_opening_angle_deg)
    rho = np.sqrt(Xb ** 2 + Zb ** 2)
    axial = np.abs(Yb)
    rho_cone = axial * np.tan(theta)                 # cone-wall radius at this height
    half_wall = lobe.wall_thickness_arcsec / 2.0
    on_wall = np.abs(rho - rho_cone) <= half_wall
    in_length = (axial >= lobe.inner_cutoff_arcsec) & (axial <= lobe.length_arcsec)
    emiss = np.where(on_wall & in_length, lobe.emissivity, 0.0)
    return emiss


def knot_emissivity(Xb, Yb, Zb, knot):
    """Two Gaussian blobs at the lobe tips, at body (0, +/- axial_position, 0)."""
    p = knot.axial_position_arcsec
    s2 = 2.0 * knot.sigma_arcsec ** 2
    r2_north = Xb ** 2 + (Yb - p) ** 2 + Zb ** 2
    r2_south = Xb ** 2 + (Yb + p) ** 2 + Zb ** 2
    emiss = knot.emissivity * (np.exp(-r2_north / s2) + np.exp(-r2_south / s2))
    # Truncate the Gaussian tails so knots stay compact (otherwise every voxel
    # carries a tiny non-zero value, which would mislabel empty space as "knots").
    r_trunc2 = (knot.truncate_nsigma * knot.sigma_arcsec) ** 2
    outside = (r2_north > r_trunc2) & (r2_south > r_trunc2)
    emiss[outside] = 0.0
    return emiss


# --- Assemble the full model ----------------------------------------------

def build_model(config=None, inclination_deg=None, position_angle_deg=None):
    """Build the Phase-1 grid model.

    Returns a dict with:
      X, Y, Z        : sky-frame coordinate cubes (arcsec)
      axis           : 1-D coordinate axis (arcsec)
      components     : {id: {"name", "emissivity"(cube), "H"(km/s/arcsec)}}
      emissivity     : combined emissivity cube
      id_cube        : integer ID of the brightest component at each voxel
      inclination_deg, position_angle_deg : orientation actually used
    """
    if config is None:
        config = cfg.DEFAULT
    if inclination_deg is None:
        inclination_deg = config.geometry.inclination_deg
    if position_angle_deg is None:
        position_angle_deg = config.geometry.position_angle_deg

    X, Y, Z, axis = make_grid(config.grid)
    Xb, Yb, Zb = sky_cubes_to_body(X, Y, Z, inclination_deg, position_angle_deg)

    # Emissivity cubes per component (body-frame shapes, sampled on sky grid).
    e_shell = shell_emissivity(Xb, Yb, Zb, config.shell)
    e_lobe = lobe_emissivity(Xb, Yb, Zb, config.lobe)
    e_knot = knot_emissivity(Xb, Yb, Zb, config.knot)

    # Homologous expansion constants H = v_char / r_char (km/s per arcsec).
    H_shell = hubble_constant(config.shell.expansion_velocity_kms,
                              config.shell.v_char_radius_arcsec)
    H_lobe = hubble_constant(config.lobe.tip_velocity_kms,
                             config.lobe.v_char_radius_arcsec)
    H_knot = hubble_constant(config.knot.velocity_kms,
                             config.knot.v_char_radius_arcsec)

    components = {
        cfg.ID_SHELL: {"name": "shell", "emissivity": e_shell, "H": H_shell},
        cfg.ID_LOBE: {"name": "lobes", "emissivity": e_lobe, "H": H_lobe},
        cfg.ID_KNOT: {"name": "knots", "emissivity": e_knot, "H": H_knot},
    }

    combined = e_shell + e_lobe + e_knot

    # ID of the brightest component at each voxel (0 where empty).
    stacked = np.stack([e_shell, e_lobe, e_knot], axis=0)
    ids = np.array([cfg.ID_SHELL, cfg.ID_LOBE, cfg.ID_KNOT])
    arg = np.argmax(stacked, axis=0)
    id_cube = np.where(combined > 0, ids[arg], 0)

    return {
        "X": X, "Y": Y, "Z": Z, "axis": axis,
        "components": components,
        "emissivity": combined,
        "id_cube": id_cube,
        "inclination_deg": inclination_deg,
        "position_angle_deg": position_angle_deg,
        "config": config,
    }
