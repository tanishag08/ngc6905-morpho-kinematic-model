"""
main.py -- Pipeline driver for the NGC 6905 morpho-kinematic model.

Run phases from the command line, e.g.:
    python main.py phase1
Each phase writes its figures/tables into outputs/ and prints a short summary.
"""

import sys
import numpy as np

import config as cfg
from orientation import draw_conventions_diagram, symmetry_axis_sky
from grid_model import build_model
from render import plot_sky_image, plot_pv
from units import kinematic_age_from_arcsec


def run_phase1(config=None):
    """Phase 1: build the analytic grid model and render first observables."""
    if config is None:
        config = cfg.DEFAULT
    np.random.seed(config.seed)
    out = config.outputs_dir

    print("=" * 70)
    print("PHASE 1: simple analytic grid model")
    print("=" * 70)

    # 0) Conventions diagram.
    p = draw_conventions_diagram(f"{out}/phase1_conventions.png")
    print(f"[saved] conventions diagram -> {p}")
    a = symmetry_axis_sky(config.geometry.inclination_deg,
                          config.geometry.position_angle_deg)
    print(f"  symmetry-axis sky unit vector (x=E, y=N, z=LOS) = "
          f"({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f})")

    # 1) Build model.
    print(f"\nBuilding {config.grid.n}^3 grid "
          f"(+/- {config.grid.half_size_arcsec:.0f} arcsec) ...")
    model = build_model(config)

    # Report homologous constants and single-age-per-component values.
    print("\nComponent homologous constants and kinematic ages")
    print(f"{'component':10s} {'H[km/s/arcsec]':>16s} "
          f"{'v_char':>8s} {'r_char':>8s} {'age[yr]':>12s}")
    d_pc = config.geometry.distance_kpc * 1000.0
    specs = {
        "shell": (config.shell.expansion_velocity_kms, config.shell.v_char_radius_arcsec),
        "lobes": (config.lobe.tip_velocity_kms, config.lobe.v_char_radius_arcsec),
        "knots": (config.knot.velocity_kms, config.knot.v_char_radius_arcsec),
    }
    for comp in model["components"].values():
        name = comp["name"]
        vchar, rchar = specs[name]
        # NOTE: this age is NOT yet deprojected for inclination (that is Phase 6);
        # it uses the characteristic radius directly as a first sanity value.
        age = kinematic_age_from_arcsec(rchar, vchar, d_pc)
        print(f"{name:10s} {comp['H']:16.3f} {vchar:8.1f} {rchar:8.1f} {age:12.0f}")

    # 2) Render sky image.
    p = plot_sky_image(model, f"{out}/phase1_sky_image.png")
    print(f"\n[saved] sky image -> {p}")

    # 3) Render P-V diagrams along the major (lobe) and minor axes.
    pa_major = config.render.pv_slit_pa_major_deg
    pa_minor = config.render.pv_slit_pa_minor_deg
    p = plot_pv(model, pa_major, f"{out}/phase1_pv_major_pa{int(pa_major)}.png",
                title=f"P-V major axis (slit PA {pa_major:.0f}°, along lobes)")
    print(f"[saved] P-V major axis -> {p}")
    p = plot_pv(model, pa_minor, f"{out}/phase1_pv_minor_pa{int(pa_minor)}.png",
                title=f"P-V minor axis (slit PA {pa_minor:.0f}°)")
    print(f"[saved] P-V minor axis -> {p}")

    # Quick numeric diagnostics.
    for cid, comp in model["components"].items():
        nz = int(np.count_nonzero(comp["emissivity"]))
        print(f"  {comp['name']:6s}: {nz:>8d} non-empty voxels, "
              f"total emissivity = {comp['emissivity'].sum():.3e}")
    print("\nPhase 1 complete.")
    return model


def run_phase2(config=None):
    """Phase 2: run the validation test suite."""
    import tests
    return tests.run_all()


PHASES = {"phase1": run_phase1, "phase2": run_phase2}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    if which not in PHASES:
        print(f"Unknown phase '{which}'. Available: {list(PHASES)}")
        sys.exit(1)
    PHASES[which]()
