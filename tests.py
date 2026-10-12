"""
tests.py -- Phase 2 validation tests for the grid model and core math.

Each test:
  * sets up a controlled scenario with a KNOWN analytic answer,
  * checks the model/pipeline against that answer with a QUANTITATIVE
    pass/fail criterion (not just eyeballing),
  * saves a diagnostic plot to outputs/,
  * returns (name, passed, message).

The seven tests mirror the project specification:
  1. Uniform thin spherical shell (homologous)      -> P-V is an ellipse.
  2. Thin ring inclined to the line of sight         -> P-V is a tilted line.
  3. Bipolar lobes at inclination 0 (in sky plane)   -> v_los symmetric about 0.
  4. Bipolar lobes at inclination 90 (along LOS)     -> overlap on sky, split in v.
  5. Flux conservation                               -> total emission vs rotation.
  6. Unit checks                                      -> 2700 AU/arcsec; 211 AU.
  7. Age sanity check                                 -> shell ~5-6 kyr.

Validated modules: orientation (rotations/projection), velocity (v_los),
grid_model (component shapes), render (P-V binning), units (conversions/age).
"""

import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from config import ID_SHELL, ID_LOBE, ID_KNOT
from orientation import body_to_sky
from velocity import hubble_constant, velocity_vectors_homologous
from grid_model import build_model, make_grid, sky_cubes_to_body
from render import _slit_coordinates
from units import arcsec_to_au, distance_travelled_au, kinematic_age_from_arcsec

OUT = cfg.DEFAULT.outputs_dir

# --- Test tolerances (documented thresholds, not physical parameters) ------
ELLIPSE_TOL_ARCSEC = 2.5     # allowed error on recovered shell radius / v-axis (~thickness + 1 cell)
LINE_R2_MIN = 0.999          # minimum R^2 for the ring's linear P-V fit
LINE_SLOPE_TOL = 0.03        # allowed fractional error on the ring P-V slope
SYMM_MEAN_TOL = 0.02         # |mean v_los| / (v scale) tolerance for symmetric case
SYMM_RATIO_TOL = 0.05        # +v vs -v weight-balance tolerance
OVERLAP_MIN = 0.80           # minimum sky-footprint overlap fraction for i=90 lobes
FLUX_TOL = 0.02              # allowed fractional spread in total flux under rotation
UNIT_RTOL = 1e-6             # relative tolerance for exact unit conversions
AGE_LO, AGE_HI = 5000.0, 6000.0  # expected shell kinematic-age band (yr)


# ===========================================================================
# Test 1 -- spherical shell -> P-V ellipse
# ===========================================================================
def test_1_spherical_shell_ellipse():
    R, t, v_exp = 25.0, 2.0, 45.0
    shell = replace(cfg.ShellParams(),
                    semi_axes_arcsec=(R, R, R), thickness_arcsec=t,
                    expansion_velocity_kms=v_exp, v_char_radius_arcsec=R)
    conf = replace(cfg.DEFAULT, shell=shell,
                   lobe=replace(cfg.LobeParams(), emissivity=0.0),
                   knot=replace(cfg.KnotParams(), emissivity=0.0))
    # A sphere is orientation-independent; use the default orientation.
    model = build_model(conf)
    comp = model["components"][ID_SHELL]
    H = comp["H"]                                   # = v_exp / R

    # Take a narrow slit through the centre at PA=0 so perp ~ 0 (pure ellipse).
    X, Y, Z = model["X"], model["Y"], model["Z"]
    pos, perp = _slit_coordinates(X, Y, pa_deg=0.0)
    e = comp["emissivity"]
    sel = (e > 0) & (np.abs(perp) <= 1.5)
    p = pos[sel]
    v_los = H * Z[sel]                              # homologous: v_los = H z

    # Envelope checks: max |v| near p=0 should be v_exp; max |p| near v=0 should be R.
    v_at_center = np.max(np.abs(v_los[np.abs(p) < 2.0]))
    p_at_vzero = np.max(np.abs(p[np.abs(v_los) < 3.0]))
    err_v = abs(v_at_center - v_exp)
    err_p = abs(p_at_vzero - R)
    passed = (err_v <= ELLIPSE_TOL_ARCSEC * (v_exp / R) + 2.0) and (err_p <= ELLIPSE_TOL_ARCSEC)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(p, v_los, s=4, alpha=0.3, color="tab:orange", label="shell voxels")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.cos(th), v_exp * np.sin(th), "b-", lw=1.5,
            label=f"ideal ellipse\n(R={R:.0f}″, v_exp={v_exp:.0f})")
    ax.set_xlabel("position along slit (arcsec)")
    ax.set_ylabel("v_los (km/s)")
    ax.set_title("Test 1: thin spherical shell -> P-V ellipse")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test1_shell_ellipse.png", dpi=130); plt.close(fig)

    msg = (f"recovered v-axis={v_at_center:.1f} (exp {v_exp:.0f}), "
           f"p-axis={p_at_vzero:.1f} (exp {R:.0f}) arcsec")
    return ("T1 spherical shell -> ellipse", passed, msg)


# ===========================================================================
# Test 2 -- inclined ring -> tilted straight line
# ===========================================================================
def test_2_inclined_ring_line():
    R, v_exp, i_ring = 25.0, 45.0, 45.0
    H = hubble_constant(v_exp, R)
    phi = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    # Ring in body x-y plane; body_to_sky with PA=0 tilts it about east (x),
    # so the line of nodes is east (x) and the projected minor axis is north (y).
    ring_body = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=1)
    ring_sky = body_to_sky(ring_body, inclination_deg=i_ring, position_angle_deg=0.0)
    vel_sky = velocity_vectors_homologous(ring_sky, H)      # v = H r (radial)
    v_los = vel_sky[:, 2]                                   # z-component

    # "Slit" along the projected minor axis (north, PA=0): position = y_sky.
    p = ring_sky[:, 1]
    # Linear fit v_los = slope * p.
    slope, intercept = np.polyfit(p, v_los, 1)
    pred = slope * p + intercept
    ss_res = np.sum((v_los - pred) ** 2)
    ss_tot = np.sum((v_los - v_los.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    slope_expected = H * np.tan(np.deg2rad(i_ring))         # = (v_exp/R) tan i
    slope_err = abs(slope - slope_expected) / abs(slope_expected)
    passed = (r2 >= LINE_R2_MIN) and (slope_err <= LINE_SLOPE_TOL)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(p, v_los, s=5, alpha=0.4, color="tab:green", label="ring points")
    xs = np.array([p.min(), p.max()])
    ax.plot(xs, slope_expected * xs, "k--", lw=1.5,
            label=f"ideal line\nslope={slope_expected:.3f} (km/s)/″")
    ax.set_xlabel("position along minor-axis slit (arcsec)")
    ax.set_ylabel("v_los (km/s)")
    ax.set_title(f"Test 2: inclined ring (i={i_ring:.0f}°) -> tilted line")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test2_ring_line.png", dpi=130); plt.close(fig)

    msg = f"R^2={r2:.5f}, slope={slope:.4f} (exp {slope_expected:.4f}), err={slope_err*100:.2f}%"
    return ("T2 inclined ring -> line", passed, msg)


# ===========================================================================
# Lobe helper: analytic cone-wall points for both lobes
# ===========================================================================
def _lobe_points(lobe, n=40000, seed=0):
    """Random points on the two hollow cone walls (body frame).

    Returns (points (N,3), weights (N,), lobe_sign (N,)) where lobe_sign is +1
    for the body +y lobe and -1 for the body -y lobe.
    """
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(lobe.half_opening_angle_deg)
    t = rng.uniform(lobe.inner_cutoff_arcsec, lobe.length_arcsec, n)   # axial distance
    phi = rng.uniform(0, 2 * np.pi, n)
    rho = t * np.tan(theta)                                            # cone-wall radius
    sign = rng.choice([1.0, -1.0], n)                                  # which lobe
    x = rho * np.cos(phi)
    y = sign * t
    z = rho * np.sin(phi)
    pts = np.stack([x, y, z], axis=1)
    w = np.full(n, lobe.emissivity)
    return pts, w, sign


# ===========================================================================
# Test 3 -- bipolar lobes at i=0 -> v_los symmetric about zero
# ===========================================================================
def test_3_lobes_inclination0():
    lobe = cfg.LobeParams()
    H = hubble_constant(lobe.tip_velocity_kms, lobe.v_char_radius_arcsec)
    pts, w, sign = _lobe_points(lobe, seed=cfg.DEFAULT.seed)
    # i=0 puts the lobe axis in the sky plane (PA chosen realistically = 160).
    sky = body_to_sky(pts, inclination_deg=0.0, position_angle_deg=160.0)
    v_los = H * sky[:, 2]

    v_scale = np.average(np.abs(v_los), weights=w)
    mean_v = np.average(v_los, weights=w)
    w_pos = w[v_los > 0].sum()
    w_neg = w[v_los < 0].sum()
    ratio = abs(w_pos - w_neg) / (w_pos + w_neg)
    passed = (abs(mean_v) / v_scale <= SYMM_MEAN_TOL) and (ratio <= SYMM_RATIO_TOL)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(v_los, bins=60, weights=w, color="tab:purple", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(mean_v, color="red", ls="--", lw=1.2, label=f"mean = {mean_v:.3f} km/s")
    ax.set_xlabel("v_los (km/s)"); ax.set_ylabel("emissivity-weighted count")
    ax.set_title("Test 3: bipolar lobes at i=0° -> v_los symmetric about 0")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test3_lobes_i0.png", dpi=130); plt.close(fig)

    msg = f"mean v_los={mean_v:.3f} km/s (scale {v_scale:.1f}), +/- weight imbalance={ratio*100:.2f}%"
    return ("T3 lobes i=0 -> symmetric v", passed, msg)


# ===========================================================================
# Test 4 -- bipolar lobes at i=90 -> overlap on sky, split in velocity
# ===========================================================================
def test_4_lobes_inclination90():
    lobe = cfg.LobeParams()
    H = hubble_constant(lobe.tip_velocity_kms, lobe.v_char_radius_arcsec)
    pts, w, sign = _lobe_points(lobe, seed=cfg.DEFAULT.seed)
    sky = body_to_sky(pts, inclination_deg=90.0, position_angle_deg=0.0)
    v_los = H * sky[:, 2]

    north = sign > 0      # body +y lobe
    south = sign < 0      # body -y lobe

    # (a) Sky-footprint overlap: bin both lobes on the same sky grid.
    extent = np.max(np.abs(sky[:, :2])) * 1.05
    bins = np.linspace(-extent, extent, 60)
    hn, _, _ = np.histogram2d(sky[north, 0], sky[north, 1], bins=[bins, bins])
    hs, _, _ = np.histogram2d(sky[south, 0], sky[south, 1], bins=[bins, bins])
    occ_n, occ_s = hn > 0, hs > 0
    overlap = np.logical_and(occ_n, occ_s).sum() / np.logical_or(occ_n, occ_s).sum()

    # (b) Velocity separation: north all receding, south all approaching.
    v_sep = v_los[north].min() - v_los[south].max()   # > 0 means a clean gap
    passed = (overlap >= OVERLAP_MIN) and (v_sep > 0)

    fig, (axs, axv) = plt.subplots(1, 2, figsize=(11, 5))
    axs.scatter(sky[north, 0], sky[north, 1], s=3, alpha=0.3, color="red", label="N lobe")
    axs.scatter(sky[south, 0], sky[south, 1], s=3, alpha=0.3, color="blue", label="S lobe")
    axs.invert_xaxis(); axs.set_aspect("equal")
    axs.set_xlabel("east (arcsec)"); axs.set_ylabel("north (arcsec)")
    axs.set_title(f"sky: lobes OVERLAP (overlap={overlap:.2f})"); axs.legend(fontsize=8)
    axv.hist(v_los[north], bins=40, weights=w[north], color="red", alpha=0.6, label="N lobe")
    axv.hist(v_los[south], bins=40, weights=w[south], color="blue", alpha=0.6, label="S lobe")
    axv.axvline(0, color="k", lw=1)
    axv.set_xlabel("v_los (km/s)"); axv.set_ylabel("count")
    axv.set_title(f"velocity: SEPARATED (gap={v_sep:.1f} km/s)"); axv.legend(fontsize=8)
    fig.suptitle("Test 4: bipolar lobes at i=90°", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test4_lobes_i90.png", dpi=130); plt.close(fig)

    msg = f"sky overlap={overlap:.2f} (>= {OVERLAP_MIN}), velocity gap={v_sep:.1f} km/s (>0)"
    return ("T4 lobes i=90 -> overlap+split", passed, msg)


# ===========================================================================
# Test 5 -- flux conservation under rotation (asymmetric filled ellipsoid)
# ===========================================================================
def _filled_ellipsoid_total(ax, ay, az, inclination_deg, position_angle_deg, grid):
    """Total emissivity of a uniform filled triaxial ellipsoid at an orientation."""
    X, Y, Z, _ = make_grid(grid)
    Xb, Yb, Zb = sky_cubes_to_body(X, Y, Z, inclination_deg, position_angle_deg)
    inside = (Xb / ax) ** 2 + (Yb / ay) ** 2 + (Zb / az) ** 2 <= 1.0
    return float(inside.sum())


def test_5_flux_conservation():
    ax, ay, az = 25.0, 15.0, 10.0           # triaxial -> rotation is non-trivial
    grid = cfg.DEFAULT.grid
    orientations = [(0, 0), (30, 45), (60, 160), (90, 90), (45, 270)]
    totals = [_filled_ellipsoid_total(ax, ay, az, i, pa, grid) for (i, pa) in orientations]
    totals = np.array(totals)
    spread = (totals.max() - totals.min()) / totals.mean()
    passed = spread <= FLUX_TOL

    fig, axp = plt.subplots(figsize=(7, 5))
    labels = [f"i={i}°\nPA={pa}°" for (i, pa) in orientations]
    axp.bar(range(len(totals)), totals, color="tab:cyan", edgecolor="k")
    axp.set_xticks(range(len(totals))); axp.set_xticklabels(labels, fontsize=8)
    axp.set_ylabel("total emissivity (voxel count)")
    axp.set_ylim(totals.mean() * 0.95, totals.mean() * 1.05)
    axp.axhline(totals.mean(), color="k", ls="--", lw=1, label=f"mean={totals.mean():.0f}")
    axp.set_title(f"Test 5: flux conservation under rotation\nspread={spread*100:.2f}% "
                  f"(filled triaxial ellipsoid)")
    axp.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test5_flux_conservation.png", dpi=130); plt.close(fig)

    msg = f"total emissivity spread across 5 orientations = {spread*100:.2f}% (tol {FLUX_TOL*100:.0f}%)"
    return ("T5 flux conservation", passed, msg)


# ===========================================================================
# Test 6 -- unit checks
# ===========================================================================
def test_6_unit_checks():
    d_pc = 2700.0
    au_per_arcsec = arcsec_to_au(1.0, d_pc)             # expect 2700 AU
    au_1kms_1000yr = distance_travelled_au(1.0, 1000.0)  # expect ~211 AU
    ok1 = abs(au_per_arcsec - 2700.0) / 2700.0 <= UNIT_RTOL
    ok2 = abs(au_1kms_1000yr - 211.0) <= 1.0            # ~211 AU within 1 AU
    passed = ok1 and ok2

    fig, axt = plt.subplots(figsize=(7, 3))
    axt.axis("off")
    rows = [
        ("1 arcsec at 2.7 kpc", f"{au_per_arcsec:.1f} AU", "2700 AU", ok1),
        ("1 km/s for 1000 yr", f"{au_1kms_1000yr:.2f} AU", "~211 AU", ok2),
    ]
    text = "Test 6: unit checks\n\n"
    for name, got, exp, ok in rows:
        text += f"  {name:24s} = {got:>12s}   (expected {exp})   {'PASS' if ok else 'FAIL'}\n"
    axt.text(0.02, 0.8, text, family="monospace", va="top", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test6_units.png", dpi=130); plt.close(fig)

    msg = f"1″@2.7kpc={au_per_arcsec:.1f} AU; 1km/s*1000yr={au_1kms_1000yr:.2f} AU"
    return ("T6 unit checks", passed, msg)


# ===========================================================================
# Test 7 -- age sanity check
# ===========================================================================
def test_7_age_sanity():
    R_arcsec, v_exp, d_pc = 19.0, 45.0, 2700.0
    age = float(kinematic_age_from_arcsec(R_arcsec, v_exp, d_pc))
    passed = AGE_LO <= age <= AGE_HI

    fig, axa = plt.subplots(figsize=(7, 3))
    axa.axvspan(AGE_LO, AGE_HI, color="tab:green", alpha=0.25, label="expected 5-6 kyr")
    axa.axvline(age, color="red", lw=2, label=f"computed = {age:.0f} yr")
    axa.set_xlim(4000, 7000); axa.set_yticks([])
    axa.set_xlabel("kinematic age (yr)")
    axa.set_title("Test 7: shell age (R=19″, d=2.7 kpc, v=45 km/s)")
    axa.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/phase2_test7_age.png", dpi=130); plt.close(fig)

    print(f"    [Test 7] computed shell kinematic age = {age:.0f} yr")
    return ("T7 age sanity (~5-6 kyr)", passed, msg_for_age(age))


def msg_for_age(age):
    return f"computed shell age = {age:.0f} yr (expected {AGE_LO:.0f}-{AGE_HI:.0f})"


# ===========================================================================
# Runner
# ===========================================================================
ALL_TESTS = [
    test_1_spherical_shell_ellipse,
    test_2_inclined_ring_line,
    test_3_lobes_inclination0,
    test_4_lobes_inclination90,
    test_5_flux_conservation,
    test_6_unit_checks,
    test_7_age_sanity,
]


CORR_MIN = 0.90              # minimum correlation between grid and particle renders


# ===========================================================================
# PHASE 3 cross-check: particle version reproduces the grid version
# ===========================================================================
def ptest_1_shell_ellipse():
    """Particle spherical shell -> P-V ellipse (same criterion as grid T1)."""
    from particles import build_particle_model
    R, t, v_exp = 25.0, 2.0, 45.0
    shell = replace(cfg.ShellParams(), semi_axes_arcsec=(R, R, R), thickness_arcsec=t,
                    expansion_velocity_kms=v_exp, v_char_radius_arcsec=R)
    conf = replace(cfg.DEFAULT, shell=shell,
                   lobe=replace(cfg.LobeParams(), emissivity=0.0),
                   knot=replace(cfg.KnotParams(), emissivity=0.0))
    pm = build_particle_model(conf, match_grid_flux=False)
    ps = pm["components"][ID_SHELL]
    x, y, z = ps.positions[:, 0], ps.positions[:, 1], ps.positions[:, 2]
    sel = np.abs(x) <= 1.5                         # narrow slit at PA=0 (perp=x)
    p = y[sel]; v = ps.v_los[sel]
    v_at_center = np.max(np.abs(v[np.abs(p) < 2.0]))
    p_at_vzero = np.max(np.abs(p[np.abs(v) < 3.0]))
    passed = (abs(v_at_center - v_exp) <= 3.0) and (abs(p_at_vzero - R) <= ELLIPSE_TOL_ARCSEC)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(p, v, s=3, alpha=0.3, color="tab:orange", label="shell particles")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.cos(th), v_exp * np.sin(th), "b-", lw=1.5, label="ideal ellipse")
    ax.set_xlabel("position along slit (arcsec)"); ax.set_ylabel("v_los (km/s)")
    ax.set_title("P-T1: PARTICLE spherical shell -> ellipse"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/phase3_ptest1_shell_ellipse.png", dpi=130); plt.close(fig)
    return ("P-T1 particle shell -> ellipse", passed,
            f"v-axis={v_at_center:.1f} (exp 45), p-axis={p_at_vzero:.1f} (exp 25)")


def ptest_3_lobes_i0():
    """Particle lobes at i=0 -> v_los symmetric about zero."""
    from particles import sample_lobes
    lobe = cfg.LobeParams()
    H = hubble_constant(lobe.tip_velocity_kms, lobe.v_char_radius_arcsec)
    rng = np.random.default_rng(cfg.DEFAULT.seed)
    pts = sample_lobes(lobe, 60000, rng)
    sky = body_to_sky(pts, inclination_deg=0.0, position_angle_deg=160.0)
    v = H * sky[:, 2]
    mean_v, scale = v.mean(), np.mean(np.abs(v))
    passed = abs(mean_v) / scale <= SYMM_MEAN_TOL
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(v, bins=60, color="tab:purple", alpha=0.8); ax.axvline(0, color="k")
    ax.set_title(f"P-T3: PARTICLE lobes i=0 (mean={mean_v:.3f} km/s)")
    ax.set_xlabel("v_los (km/s)")
    fig.tight_layout(); fig.savefig(f"{OUT}/phase3_ptest3_lobes_i0.png", dpi=130); plt.close(fig)
    return ("P-T3 particle lobes i=0", passed, f"mean v_los={mean_v:.3f} km/s (scale {scale:.1f})")


def ptest_4_lobes_i90():
    """Particle lobes at i=90 -> overlap on sky, split in velocity."""
    from particles import sample_lobes
    lobe = cfg.LobeParams()
    H = hubble_constant(lobe.tip_velocity_kms, lobe.v_char_radius_arcsec)
    rng = np.random.default_rng(cfg.DEFAULT.seed)
    # sample_lobes assigns lobe sign internally; recover it from body-y sign.
    pts = sample_lobes(lobe, 60000, rng)
    north = pts[:, 1] > 0
    sky = body_to_sky(pts, inclination_deg=90.0, position_angle_deg=0.0)
    v = H * sky[:, 2]
    ext = np.max(np.abs(sky[:, :2])) * 1.05
    bins = np.linspace(-ext, ext, 60)
    hn, _, _ = np.histogram2d(sky[north, 0], sky[north, 1], bins=[bins, bins])
    hs, _, _ = np.histogram2d(sky[~north, 0], sky[~north, 1], bins=[bins, bins])
    overlap = np.logical_and(hn > 0, hs > 0).sum() / np.logical_or(hn > 0, hs > 0).sum()
    v_sep = v[north].min() - v[~north].max()
    passed = (overlap >= OVERLAP_MIN) and (v_sep > 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.scatter(sky[north, 0], sky[north, 1], s=2, alpha=0.3, color="red")
    a1.scatter(sky[~north, 0], sky[~north, 1], s=2, alpha=0.3, color="blue")
    a1.invert_xaxis(); a1.set_aspect("equal"); a1.set_title(f"sky overlap={overlap:.2f}")
    a2.hist(v[north], bins=40, color="red", alpha=0.6)
    a2.hist(v[~north], bins=40, color="blue", alpha=0.6); a2.axvline(0, color="k")
    a2.set_title(f"velocity split (gap={v_sep:.1f})"); a2.set_xlabel("v_los (km/s)")
    fig.suptitle("P-T4: PARTICLE lobes at i=90°", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/phase3_ptest4_lobes_i90.png", dpi=130); plt.close(fig)
    return ("P-T4 particle lobes i=90", passed, f"overlap={overlap:.2f}, gap={v_sep:.1f} km/s")


def ptest_5_flux_conservation():
    """Particle flux is exactly conserved under rotation (count/weights invariant)."""
    from particles import build_particle_model
    totals = []
    half = cfg.DEFAULT.grid.half_size_arcsec
    for (i, pa) in [(0, 0), (30, 45), (60, 160), (90, 90), (45, 270)]:
        pm = build_particle_model(cfg.DEFAULT, inclination_deg=i, position_angle_deg=pa,
                                  n_per_component=50000, match_grid_flux=False)
        pts = pm["particles"].positions
        w = pm["particles"].weights
        infov = (np.abs(pts[:, 0]) <= half) & (np.abs(pts[:, 1]) <= half)
        totals.append(w[infov].sum())
    totals = np.array(totals)
    spread = (totals.max() - totals.min()) / totals.mean()
    passed = spread <= FLUX_TOL
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(5), totals, color="tab:cyan", edgecolor="k")
    ax.set_ylim(totals.mean() * 0.99, totals.mean() * 1.01)
    ax.set_title(f"P-T5: PARTICLE flux conservation (spread={spread*100:.3f}%)")
    ax.set_ylabel("total weight in FOV")
    fig.tight_layout(); fig.savefig(f"{OUT}/phase3_ptest5_flux.png", dpi=130); plt.close(fig)
    return ("P-T5 particle flux conservation", passed, f"spread={spread*100:.3f}% (tol {FLUX_TOL*100:.0f}%)")


def ptest_6_grid_vs_particle():
    """Grid vs particle renders of the DEFAULT model must agree (correlation)."""
    from particles import build_particle_model
    from render import pv_diagram, particle_pv_diagram, sky_image, particle_sky_image
    gm = build_model(cfg.DEFAULT)
    pm = build_particle_model(cfg.DEFAULT, match_grid_flux=True)

    pa = cfg.DEFAULT.render.pv_slit_pa_major_deg
    pv_g, pe, ve = pv_diagram(gm, pa)
    pv_p, _, _ = particle_pv_diagram(pm, pa)
    img_g, half = sky_image(gm)
    img_p, _ = particle_sky_image(pm)

    from scipy.ndimage import gaussian_filter

    def corr(a, b, smooth=0.0):
        # Compare the underlying distributions: a light smoothing removes the
        # particle Poisson noise in the thin slit so we are not penalised for
        # sub-pixel placement of thin features (grid also discretises).
        if smooth > 0:
            a = gaussian_filter(a, smooth)
            b = gaussian_filter(b, smooth)
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    # The sky image is well sampled -> compare directly. The thin-slit P-V is
    # sparse for particles -> compare with a 1.5-pixel smoothing.
    c_img = corr(img_g, img_p, smooth=0.0)
    c_pv = corr(pv_g, pv_p, smooth=1.5)
    passed = (c_pv >= CORR_MIN) and (c_img >= CORR_MIN)

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for ax, data, ttl in [(axs[0, 0], img_g, "GRID sky image"),
                          (axs[0, 1], img_p, "PARTICLE sky image")]:
        ax.imshow(data, origin="lower", extent=[-half, half, -half, half], cmap="inferno")
        ax.invert_xaxis(); ax.set_title(ttl); ax.set_xlabel("E (arcsec)"); ax.set_ylabel("N (arcsec)")
    for ax, data, ttl in [(axs[1, 0], pv_g, f"GRID P-V (PA {pa:.0f}°)"),
                          (axs[1, 1], pv_p, f"PARTICLE P-V (PA {pa:.0f}°)")]:
        ax.imshow(data, origin="lower", aspect="auto", cmap="inferno",
                  extent=[pe[0], pe[-1], ve[0], ve[-1]])
        ax.set_title(ttl); ax.set_xlabel("position (arcsec)"); ax.set_ylabel("v_los (km/s)")
    fig.suptitle(f"P-T6: grid vs particle  (image corr={c_img:.3f}, P-V corr={c_pv:.3f})",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/phase3_ptest6_grid_vs_particle.png", dpi=130); plt.close(fig)
    return ("P-T6 grid vs particle", passed, f"image corr={c_img:.3f}, P-V corr={c_pv:.3f} (min {CORR_MIN})")


PARTICLE_TESTS = [ptest_1_shell_ellipse, ptest_3_lobes_i0, ptest_4_lobes_i90,
                  ptest_5_flux_conservation, ptest_6_grid_vs_particle]


def run_particle_crosscheck():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("PHASE 3 cross-check: particle model vs grid model")
    print("=" * 70)
    results = []
    for fn in PARTICLE_TESTS:
        name, passed, msg = fn()
        results.append((name, passed, msg))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:30s} | {msg}")
    n_pass = sum(1 for _, p, _ in results if p)
    print("-" * 70)
    print(f"  {n_pass}/{len(results)} particle cross-checks passed.")
    return results


def run_all():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 70)
    print("PHASE 2: validation tests")
    print("=" * 70)
    results = []
    for fn in ALL_TESTS:
        name, passed, msg = fn()
        results.append((name, passed, msg))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:34s} | {msg}")

    n_pass = sum(1 for _, p, _ in results if p)
    print("-" * 70)
    print(f"  {n_pass}/{len(results)} tests passed.")

    # Summary figure.
    fig, axs = plt.subplots(figsize=(9, 0.5 * len(results) + 1.5))
    axs.axis("off")
    y = 1.0
    axs.text(0.0, y, "Phase 2 validation summary", fontsize=13, fontweight="bold")
    y -= 0.12
    for name, passed, msg in results:
        color = "green" if passed else "red"
        axs.text(0.0, y, f"[{'PASS' if passed else 'FAIL'}]", color=color,
                 family="monospace", fontweight="bold")
        axs.text(0.12, y, f"{name}", family="monospace")
        axs.text(0.55, y, msg, family="monospace", fontsize=7.5)
        y -= 0.11
    fig.savefig(f"{OUT}/phase2_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


if __name__ == "__main__":
    run_all()
