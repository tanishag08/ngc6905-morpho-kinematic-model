"""
velocity.py -- Velocity laws for the morpho-kinematic model.

HOMOLOGOUS ("Hubble-type") EXPANSION -------------------------------------

The default velocity field is homologous: every parcel moves radially outward
from the central star with a speed proportional to its distance r from the star

    v_vec = H * r_vec            (H is a constant, units km/s per arcsec here)

This is the kinematic signature of a freely coasting, ballistic outflow: if all
material left the star at the same time t_0 and has coasted at constant speed,
then r = v * (t - t_0) for every parcel, so
    v = r / (t - t_0)  =>  v / r = 1 / (t - t_0) = H = constant.
Hence a single H describes the whole structure and, crucially,
    kinematic age = r / v = 1 / H  is THE SAME for every parcel.
=> Homologous expansion means a component has a SINGLE well-defined kinematic age.

We allow each component its own H so that different ejection episodes (shell,
lobes, knots) can have different ages. H is set so the component reaches its
measured/assumed characteristic velocity v_char at its characteristic radius
r_char:
    H = v_char / r_char.

LINE-OF-SIGHT VELOCITY ----------------------------------------------------

Because v_vec = H * r_vec is purely radial, its component along the line of
sight (sky z-axis) is simply
    v_los = v_z = H * z
where z is the sky-frame z-coordinate of the parcel (in arcsec). Positive z is
away from the observer, so positive v_los is a redshift -- consistent with
orientation.py. This makes rendering P-V diagrams very cheap.

GENERALISED POWER LAW -----------------------------------------------------

Optionally a component may follow v = k * r^alpha (radial). alpha = 1 is the
homologous case. alpha != 1 matters when the outflow is NOT freely coasting --
e.g. if the lobes are still being accelerated (alpha > 1 gives faster tips,
"Hubble-like but steeper") or decelerated by swept-up material (alpha < 1).
For a power law the simple "single age = 1/H" statement no longer holds; the
age varies with radius. We keep homologous (alpha = 1) as the default.
"""

import numpy as np


def hubble_constant(v_char_kms, r_char_arcsec):
    """Return the homologous expansion constant H = v_char / r_char.

    Units: km/s per arcsec. (We keep radius in arcsec for the velocity FIELD;
    conversion to physical units happens only when computing ages.)
    """
    return v_char_kms / r_char_arcsec


def velocity_vectors_homologous(points_arcsec, H):
    """Homologous velocity vectors v = H * r for points (N,3) in arcsec.

    Returns an (N,3) array of velocities in km/s.
    """
    pts = np.asarray(points_arcsec, dtype=float)
    return H * pts


def velocity_vectors_powerlaw(points_arcsec, k, alpha):
    """Radial power-law velocity v = k * r^alpha.

    Direction is radial (r_vec / |r_vec|); magnitude is k * |r|^alpha.
    alpha = 1 reduces to homologous with H = k.
    """
    pts = np.asarray(points_arcsec, dtype=float)
    r = np.linalg.norm(pts, axis=-1, keepdims=True)
    # Avoid division by zero at the origin.
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(r > 0, pts / r, 0.0)
    speed = k * np.power(np.maximum(r[..., 0], 0.0), alpha)
    return unit * speed[..., None]


def v_los_homologous(z_arcsec, H):
    """Line-of-sight velocity for homologous expansion: v_los = H * z.

    z_arcsec is the sky-frame z-coordinate (arcsec); returns km/s.
    """
    return H * np.asarray(z_arcsec, dtype=float)
