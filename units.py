"""
units.py -- Unit conversions and physical constants for the NGC 6905 model.

All physical constants are defined ONCE here so there are no magic numbers
scattered through the code. Everything is commented with its unit.

Coordinate/length conventions used throughout the project:
  - Angular sizes on the sky are in arcseconds (arcsec).
  - Physical (linear) sizes are in astronomical units (AU) or parsecs (pc).
  - Velocities are in kilometres per second (km/s).
  - Times / ages are in years (yr).

The fundamental astronomical relation we lean on is the *parallax* definition:
    1 AU seen from a distance of 1 pc subtends exactly 1 arcsec.
Therefore a physical length L (in AU) at distance d (in pc) subtends an angle
    theta (arcsec) = L (AU) / d (pc).
Equivalently a measured angle theta at distance d corresponds to
    L (AU) = theta (arcsec) * d (pc).
So at d = 2700 pc, 1 arcsec corresponds to 2700 AU (used in validation test 6).
"""

import numpy as np

# --- Fundamental constants (SI-ish, but mixed astronomical units) ---------

KM_PER_AU = 1.495978707e8      # kilometres in one astronomical unit (IAU 2012 exact)
AU_PER_PC = 206264.806247      # astronomical units in one parsec (= arcsec per radian)
PC_PER_KPC = 1000.0            # parsecs per kiloparsec
SECONDS_PER_YEAR = 3.15576e7   # seconds in a Julian year (365.25 days)

# Convenience: kilometres in one parsec
KM_PER_PC = KM_PER_AU * AU_PER_PC


# --- Angle <-> physical length conversions --------------------------------

def arcsec_to_au(theta_arcsec, distance_pc):
    """Convert an angular size (arcsec) to a physical length (AU) at a distance.

    Uses L[AU] = theta[arcsec] * d[pc]  (parallax relation).
    """
    return np.asarray(theta_arcsec) * distance_pc


def au_to_arcsec(length_au, distance_pc):
    """Convert a physical length (AU) to an angular size (arcsec) at a distance."""
    return np.asarray(length_au) / distance_pc


def arcsec_to_pc(theta_arcsec, distance_pc):
    """Convert an angular size (arcsec) to a physical length in parsecs."""
    return arcsec_to_au(theta_arcsec, distance_pc) / AU_PER_PC


def kpc_to_pc(distance_kpc):
    """Kiloparsecs -> parsecs."""
    return distance_kpc * PC_PER_KPC


# --- Kinematic age --------------------------------------------------------

def kinematic_age_years(radius_au, velocity_kms):
    """Kinematic age t = r / v, returned in years.

    r is a physical radius in AU, v an expansion speed in km/s.

    Derivation of the unit factor:
        t[s]  = r[AU] * KM_PER_AU[km/AU] / v[km/s]
        t[yr] = t[s] / SECONDS_PER_YEAR

    Sanity: v = 1 km/s sustained for 1000 yr travels
        1 km/s * 1000 yr * SECONDS_PER_YEAR = 3.15576e10 km = 211 AU
    (validation test 6).
    """
    radius_au = np.asarray(radius_au, dtype=float)
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    time_seconds = radius_au * KM_PER_AU / velocity_kms
    return time_seconds / SECONDS_PER_YEAR


def kinematic_age_from_arcsec(radius_arcsec, velocity_kms, distance_pc):
    """Convenience wrapper: kinematic age (yr) from an angular radius (arcsec).

    First deprojection-free conversion arcsec -> AU, then r/v.
    (Deprojection by inclination is handled by the caller in ages.py.)
    """
    radius_au = arcsec_to_au(radius_arcsec, distance_pc)
    return kinematic_age_years(radius_au, velocity_kms)


def distance_travelled_au(velocity_kms, time_years):
    """Distance (AU) travelled at constant speed v (km/s) over t (yr).

    Used by validation test 6 (expect 211 AU for 1 km/s over 1000 yr).
    """
    distance_km = velocity_kms * time_years * SECONDS_PER_YEAR
    return distance_km / KM_PER_AU
