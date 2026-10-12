"""
config.py -- All model parameters for NGC 6905 in one place.

RULES obeyed here:
  * No magic numbers live outside this file.
  * Every value carries its unit in a comment.
  * Every ASSUMPTION (anything not from a published measurement) is labelled
    with "ASSUMPTION" so it is obvious in code, plots, and the summary.
  * Published values carry a short source tag. They must never be changed
    silently; if a fit moves one, that change is reported explicitly.

Parameters are grouped into dataclasses so individual components can be
constructed and tested in isolation.

References for published values:
  Distance        : Bailer-Jones et al. 2021 (Gaia EDR3), 2.7 +/- 0.2 kpc
  Inclination/PA  : Cuesta, Phillips & Mampaso 1993
  Shell v_exp     : ~45 km/s [O III] (see note below)
  Systemic v      : ~ -8 km/s heliocentric

Note on the shell expansion velocity: the 1993 study quotes ~100 km/s, but
that is almost certainly the full velocity *splitting* (2 x v_exp) seen in a
P-V line, so the true expansion velocity is ~45-50 km/s. We adopt 45 km/s.
"""

from dataclasses import dataclass, field, asdict

# Component integer IDs (used to tag voxels / particles by component).
ID_SHELL = 1
ID_LOBE = 2
ID_KNOT = 3

COMPONENT_NAMES = {ID_SHELL: "shell", ID_LOBE: "lobes", ID_KNOT: "knots"}


# ---------------------------------------------------------------------------
# Central ellipsoidal shell
# ---------------------------------------------------------------------------
@dataclass
class ShellParams:
    # Semi-axes (a_x, a_y, a_z) of the ellipsoid in arcsec, in the BODY frame.
    # Body +y is the nebular symmetry (lobe) axis. The shell is roundish, so
    # we start spherical at the published angular radius ~19 arcsec.
    # (Literature also quotes a 47 x 34 arcsec inner shell = radii ~23 x 17;
    #  roundish 19 is our default, adjustable here.)       [PUBLISHED ~19 arcsec]
    semi_axes_arcsec: tuple = (19.0, 19.0, 19.0)
    thickness_arcsec: float = 4.0            # shell wall thickness, arcsec  [ASSUMPTION - from image]
    expansion_velocity_kms: float = 45.0     # [O III] expansion speed, km/s [PUBLISHED]
    v_char_radius_arcsec: float = 19.0       # radius at which v_exp is reached, arcsec [PUBLISHED]
    emissivity: float = 1.0                  # relative surface brightness weight


# ---------------------------------------------------------------------------
# Bipolar lobes: two opposing hollow cones along body +y / -y
# ---------------------------------------------------------------------------
@dataclass
class LobeParams:
    length_arcsec: float = 40.0              # each lobe half-length, arcsec (total extent ~80) [PUBLISHED total ~80 arcsec]
    half_opening_angle_deg: float = 18.0     # cone half-opening angle, deg  [ASSUMPTION - from image]
    wall_thickness_arcsec: float = 4.0       # cone wall thickness, arcsec   [ASSUMPTION - from image]
    inner_cutoff_arcsec: float = 5.0         # inner radius where lobes start, arcsec [ASSUMPTION]
    tip_velocity_kms: float = 60.0           # expansion speed at lobe tip, km/s [ASSUMPTION - free parameter]
    v_char_radius_arcsec: float = 40.0       # radius at which tip velocity is reached, arcsec
    emissivity: float = 0.7                  # relative surface brightness weight


# ---------------------------------------------------------------------------
# Low-ionization knots (ansae) at the lobe tips: two Gaussian blobs
# ---------------------------------------------------------------------------
@dataclass
class KnotParams:
    axial_position_arcsec: float = 40.0      # distance of each knot from star along axis, arcsec [ASSUMPTION ~ lobe tip]
    sigma_arcsec: float = 4.0                # Gaussian size (std dev), arcsec [ASSUMPTION - from image]
    truncate_nsigma: float = 4.0             # zero the Gaussian beyond this many sigma (keeps knots compact)
    velocity_kms: float = 70.0               # knot expansion speed, km/s   [ASSUMPTION - free parameter]
    v_char_radius_arcsec: float = 40.0       # radius at which knot velocity is reached, arcsec
    emissivity: float = 2.0                  # relative surface brightness weight (knots are bright in [N II])


# ---------------------------------------------------------------------------
# Global geometry / orientation
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    distance_kpc: float = 2.7                # Gaia distance, kpc           [PUBLISHED]
    distance_err_kpc: float = 0.2            # 1-sigma distance error, kpc  [PUBLISHED]
    inclination_deg: float = 60.0            # lobe axis vs plane of sky, deg [PUBLISHED]
    position_angle_deg: float = 160.0        # PA of lobe axis, N through E, deg [PUBLISHED]
    systemic_velocity_kms: float = -8.0      # heliocentric systemic velocity, km/s [PUBLISHED]


# ---------------------------------------------------------------------------
# Emissivity grid (Phase 1)
# ---------------------------------------------------------------------------
@dataclass
class GridParams:
    n: int = 128                             # grid cells per axis (n^3 total)
    half_size_arcsec: float = 50.0           # grid spans [-half, +half] arcsec on each axis


# ---------------------------------------------------------------------------
# Rendering (sky image + P-V diagrams). Phase-4 fields included for continuity.
# ---------------------------------------------------------------------------
@dataclass
class RenderParams:
    pixel_scale_arcsec: float = 0.5          # synthetic image pixel scale, arcsec/pix
    seeing_fwhm_arcsec: float = 1.0          # Gaussian seeing FWHM, arcsec (Phase 4)
    # Long-slit P-V diagram settings:
    pv_slit_pa_major_deg: float = 160.0      # slit along the lobe axis, deg [PUBLISHED PA]
    pv_slit_pa_minor_deg: float = 70.0       # perpendicular slit, deg
    pv_slit_width_arcsec: float = 2.0        # slit width, arcsec
    pv_v_min_kms: float = -150.0             # P-V velocity axis minimum, km/s
    pv_v_max_kms: float = 150.0              # P-V velocity axis maximum, km/s
    pv_v_bin_kms: float = 2.0                # P-V velocity bin width, km/s
    pv_instrumental_fwhm_kms: float = 10.0   # instrumental velocity smoothing FWHM (Phase 4)
    # Channel maps (Phase 4):
    channel_width_kms: float = 10.0
    channel_v_min_kms: float = -100.0
    channel_v_max_kms: float = 100.0


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 42                           # fixed RNG seed for reproducibility
    n_particles_per_component: int = 200000  # Phase 3 particle count
    outputs_dir: str = "outputs"
    shell: ShellParams = field(default_factory=ShellParams)
    lobe: LobeParams = field(default_factory=LobeParams)
    knot: KnotParams = field(default_factory=KnotParams)
    geometry: Geometry = field(default_factory=Geometry)
    grid: GridParams = field(default_factory=GridParams)
    render: RenderParams = field(default_factory=RenderParams)

    def to_dict(self):
        """Full config as a plain nested dict (for JSON save/load)."""
        return asdict(self)


# A ready-to-use default configuration instance.
DEFAULT = Config()
