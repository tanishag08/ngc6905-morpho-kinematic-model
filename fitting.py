"""
fitting.py -- Image-based parameter fitting.

We optimise the GEOMETRY parameters (inclination, position angle, shell radius,
lobe length, lobe opening angle) to make the synthetic sky image match a target
image, using scipy.optimize. The loss is 1 - Pearson correlation of the two
normalised, seeing-matched images (correlation is insensitive to the unknown
overall brightness scaling).

IMPORTANT PHYSICS CAVEAT (stated clearly, per the spec):
  An image constrains only the projected GEOMETRY. It carries NO velocity
  information, so image-only fitting CANNOT constrain the expansion velocities
  or (therefore) the kinematic ages. Velocities remain fixed at their
  published/assumed values; only spectroscopy (P-V data) could constrain them.

To keep the optimiser fast and its loss deterministic we render with a reduced
particle count, a fixed RNG seed, and match_grid_flux=False.

Published starting values are never overwritten silently: fit_report() prints
the best-fit values AND their offsets from the published starting point.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

import config as cfg
from particles import build_particle_model
from render import synthetic_image

# Parameter order used by the optimiser.
PARAM_NAMES = ["inclination_deg", "position_angle_deg", "shell_radius_arcsec",
               "lobe_length_arcsec", "lobe_opening_deg"]

# Which of the fitted parameters are published (vs assumptions). Used only for
# reporting how far a fit moved a published value.
PUBLISHED_FLAG = {"inclination_deg": True, "position_angle_deg": True,
                  "shell_radius_arcsec": True, "lobe_length_arcsec": True,
                  "lobe_opening_deg": False}

# Fit settings.
FIT_N_PARTICLES = 60000      # per component while fitting (speed vs noise)
FIT_PIXEL_SCALE = 1.0        # arcsec/pix for the fit image
FIT_SEEING_FWHM = 2.0        # arcsec; extra smoothing stabilises the loss


def params_to_config(params, base_config):
    """Build a Config from a parameter vector (keeps velocities at base values)."""
    inc, pa, shell_r, lobe_len, open_deg = params
    shell = replace(base_config.shell,
                    semi_axes_arcsec=(shell_r, shell_r, shell_r),
                    v_char_radius_arcsec=shell_r)
    lobe = replace(base_config.lobe, length_arcsec=lobe_len,
                   half_opening_angle_deg=open_deg, v_char_radius_arcsec=lobe_len)
    knot = replace(base_config.knot, axial_position_arcsec=lobe_len,
                   v_char_radius_arcsec=lobe_len)
    return replace(base_config, shell=shell, lobe=lobe, knot=knot), inc, pa


def model_image(params, base_config, half_arcsec, pixel_scale=FIT_PIXEL_SCALE,
                seeing=FIT_SEEING_FWHM, n_particles=FIT_N_PARTICLES):
    """Render the synthetic image for a parameter vector on the fit grid."""
    conf, inc, pa = params_to_config(params, base_config)
    pm = build_particle_model(conf, inclination_deg=inc, position_angle_deg=pa,
                              n_per_component=n_particles, match_grid_flux=False)
    img, _ = synthetic_image(pm, pixel_scale=pixel_scale, seeing_fwhm=seeing,
                             fov_arcsec=2 * half_arcsec)
    return img


def _normalise(a):
    a = np.nan_to_num(np.asarray(a, dtype=float))
    s = a.sum()
    return a / s if s > 0 else a


def image_loss(params, target_img, base_config, half_arcsec, bounds):
    """1 - correlation between model and target images, with a bounds penalty."""
    # Soft bounds: reject out-of-range parameters with a large penalty.
    for val, (lo, hi) in zip(params, bounds):
        if val < lo or val > hi:
            return 10.0
    model = model_image(params, base_config, half_arcsec)
    a, b = _normalise(model).ravel(), _normalise(target_img).ravel()
    if a.std() == 0 or b.std() == 0:
        return 10.0
    corr = np.corrcoef(a, b)[0, 1]
    return 1.0 - corr


def fit_image(target_img, half_arcsec, base_config=None, start_params=None,
              bounds=None, verbose=True):
    """Fit geometry parameters to a target image. Returns a result dict."""
    if base_config is None:
        base_config = cfg.DEFAULT
    if start_params is None:
        g = base_config.geometry
        start_params = [g.inclination_deg, g.position_angle_deg,
                        base_config.shell.semi_axes_arcsec[0],
                        base_config.lobe.length_arcsec,
                        base_config.lobe.half_opening_angle_deg]
    if bounds is None:
        bounds = [(10, 90), (120, 200), (10, 30), (25, 50), (8, 35)]

    start = np.array(start_params, dtype=float)
    res = minimize(image_loss, start,
                   args=(target_img, base_config, half_arcsec, bounds),
                   method="Powell",
                   options={"xtol": 1e-2, "ftol": 1e-3, "maxiter": 300})
    best = res.x
    out = {"start": dict(zip(PARAM_NAMES, start)),
           "best": dict(zip(PARAM_NAMES, best)),
           "start_loss": image_loss(start, target_img, base_config, half_arcsec, bounds),
           "best_loss": float(res.fun),
           "success": bool(res.success), "result": res}
    if verbose:
        fit_report(out)
    return out


def fit_report(out):
    """Print best-fit values and their offsets from the (published) start."""
    print("\n  Image fit results (geometry only; velocities NOT constrained):")
    print(f"  {'parameter':24s} {'start':>10s} {'best':>10s} {'delta':>10s}  source")
    for name in PARAM_NAMES:
        s, b = out["start"][name], out["best"][name]
        tag = "PUBLISHED" if PUBLISHED_FLAG[name] else "assumption"
        print(f"  {name:24s} {s:10.2f} {b:10.2f} {b - s:+10.2f}  {tag}")
    print(f"  loss (1 - corr): start={out['start_loss']:.4f} -> best={out['best_loss']:.4f}")
    moved = [n for n in PARAM_NAMES if PUBLISHED_FLAG[n]
             and abs(out['best'][n] - out['start'][n]) > 1e-6]
    if moved:
        print("  NOTE: the fit moved these PUBLISHED values (reported, not silent):")
        for n in moved:
            print(f"        {n}: {out['start'][n]:.2f} -> {out['best'][n]:.2f}")


def self_consistency_demo(base_config=None, true_params=None, seed=123):
    """Generate a target from known params, then recover them by fitting.

    This validates the fitter without needing a network download: it should
    recover parameters close to the injected truth from the image alone.
    Returns (out, true_params, target_img, half).
    """
    if base_config is None:
        base_config = cfg.DEFAULT
    half = base_config.grid.half_size_arcsec
    if true_params is None:
        # Deliberately offset from the published start so recovery is meaningful.
        true_params = [50.0, 150.0, 22.0, 44.0, 22.0]
    # Build the target with MORE particles + seeing so it is a clean target.
    target = model_image(true_params, base_config, half, n_particles=150000)
    out = fit_image(target, half, base_config=base_config, verbose=False)
    out["true"] = dict(zip(PARAM_NAMES, true_params))
    return out, true_params, target, half
