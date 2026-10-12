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


def _modifier_demo(config, save_path):
    """Demonstrate the geometric modifiers on a particle sphere (before/after)."""
    import matplotlib.pyplot as plt
    from particles import sample_shell, ParticleSet
    import config as cfg
    from modifiers import Twist, Bump, Taper
    from dataclasses import replace

    rng = np.random.default_rng(config.seed)
    # Sphere reveals Bump and Taper; a flattened ellipsoid is needed to SEE Twist
    # (a sphere is rotationally symmetric, so twisting it changes nothing).
    sphere_shell = replace(cfg.ShellParams(), semi_axes_arcsec=(20, 20, 20), thickness_arcsec=1.5)
    flat_shell = replace(cfg.ShellParams(), semi_axes_arcsec=(22, 22, 6), thickness_arcsec=1.5)
    pts = sample_shell(sphere_shell, 8000, rng)
    flat = sample_shell(flat_shell, 8000, rng)

    demos = [("original sphere", pts),
             ("Twist 8°/″ (flattened\nellipsoid, axis y)", Twist(1, 8.0)(flat)),
             ("Bump (axis y, amp 3, freq 6)", Bump(1, 3.0, 6)(pts)),
             ("Taper (axis y, 1.0->0.2)", Taper(1, 1.0, 0.2, use_abs=True)(pts))]
    fig = plt.figure(figsize=(14, 4))
    for idx, (ttl, P) in enumerate(demos):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.3, color="tab:blue")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlim(-25, 25); ax.set_ylim(-25, 25); ax.set_zlim(-25, 25)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("Phase 3: SHAPE-style modifiers on a particle sphere", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def run_phase3(config=None):
    """Phase 3: SHAPE-style meshes, modifiers, particles; cross-check vs grid."""
    import tests
    from particles import build_particle_model
    from render import plot_particle_sky_image, plot_particle_pv
    from meshes import component_meshes, preview_meshes

    if config is None:
        config = cfg.DEFAULT
    np.random.seed(config.seed)
    out = config.outputs_dir

    print("=" * 70)
    print("PHASE 3: SHAPE-style mesh + particle model")
    print("=" * 70)

    # 1) Component meshes preview.
    meshes = component_meshes(config)
    watertight = {k: m.is_watertight for k, m in meshes.items()}
    print("Component meshes (body frame):")
    for k, m in meshes.items():
        print(f"  {k:7s}: {len(m.vertices):5d} verts, {len(m.faces):5d} faces, "
              f"watertight={m.is_watertight}")
    p = preview_meshes(meshes, f"{out}/phase3_meshes_preview.png")
    print(f"[saved] mesh preview -> {p}")

    # 2) Modifier demonstration.
    p = _modifier_demo(config, f"{out}/phase3_modifier_demo.png")
    print(f"[saved] modifier demo -> {p}")

    # 3) Build particle model and render.
    print(f"\nBuilding particle model "
          f"({config.n_particles_per_component} particles/component)...")
    pm = build_particle_model(config)
    print(f"  total particles = {len(pm['particles'])}")
    p = plot_particle_sky_image(pm, f"{out}/phase3_particle_sky_image.png",
                                title="NGC 6905 particle sky image")
    print(f"[saved] particle sky image -> {p}")
    pa_major = config.render.pv_slit_pa_major_deg
    pa_minor = config.render.pv_slit_pa_minor_deg
    plot_particle_pv(pm, pa_major, f"{out}/phase3_particle_pv_major.png",
                     title=f"Particle P-V major axis (PA {pa_major:.0f}°)")
    plot_particle_pv(pm, pa_minor, f"{out}/phase3_particle_pv_minor.png",
                     title=f"Particle P-V minor axis (PA {pa_minor:.0f}°)")
    print(f"[saved] particle P-V diagrams (PA {pa_major:.0f}°, {pa_minor:.0f}°)")

    # 4) Cross-check against the grid model.
    print()
    tests.run_particle_crosscheck()
    print("\nPhase 3 complete.")
    return pm


def _plot_fit_demo(out, true_params, target, half, save_path):
    """Show the self-consistency fit: target, best-fit model, difference."""
    import matplotlib.pyplot as plt
    import fitting
    best = [out["best"][n] for n in fitting.PARAM_NAMES]
    model = fitting.model_image(best, cfg.DEFAULT, half, n_particles=150000)

    def norm(a):
        a = np.nan_to_num(a); return a / a.max() if a.max() > 0 else a
    t, m = norm(target), norm(model)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, ttl, cmap in [(axes[0], t, "injected target", "inferno"),
                               (axes[1], m, "best-fit model", "inferno"),
                               (axes[2], m - t, "model - target", "coolwarm")]:
        im = ax.imshow(img, origin="lower", extent=[-half, half, -half, half],
                       cmap=cmap, vmin=(-1 if cmap == "coolwarm" else 0), vmax=1)
        ax.invert_xaxis(); ax.set_xlabel("E (arcsec)"); ax.set_ylabel("N (arcsec)")
        ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Phase 4 fit self-consistency (image-only; velocities not fitted)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path


def run_phase4(config=None):
    """Phase 4: realistic synthetic obs, real image, and image-based fitting."""
    import fitting
    import observations as obs
    from particles import build_particle_model
    from render import (plot_synthetic_image, plot_composite_image,
                        plot_channel_maps, plot_synthetic_pv, synthetic_image)

    if config is None:
        config = cfg.DEFAULT
    np.random.seed(config.seed)
    out = config.outputs_dir
    print("=" * 70)
    print("PHASE 4: synthetic observations, real image, and fitting")
    print("=" * 70)

    pm = build_particle_model(config)

    # 1) Realistic synthetic outputs.
    p = plot_synthetic_image(pm, f"{out}/phase4_synthetic_image.png")
    print(f"[saved] synthetic image (seeing) -> {p}")
    p = plot_composite_image(pm, f"{out}/phase4_composite.png")
    print(f"[saved] colour composite -> {p}")
    pa_major = config.render.pv_slit_pa_major_deg
    pa_minor = config.render.pv_slit_pa_minor_deg
    plot_synthetic_pv(pm, pa_major, f"{out}/phase4_pv_major.png",
                      title=f"Synthetic P-V major axis (PA {pa_major:.0f}°, smoothed)")
    plot_synthetic_pv(pm, pa_minor, f"{out}/phase4_pv_minor.png",
                      title=f"Synthetic P-V minor axis (PA {pa_minor:.0f}°, smoothed)")
    print(f"[saved] smoothed P-V diagrams (PA {pa_major:.0f}°, {pa_minor:.0f}°)")
    p = plot_channel_maps(pm, f"{out}/phase4_channel_maps.png")
    print(f"[saved] channel maps -> {p}")

    # 2) Real image: prefer the saved Pan-STARRS cutout (self-contained); fall
    #    back to a SkyView download; otherwise skip gracefully.
    import os
    half = config.grid.half_size_arcsec
    real_disp = real_fit = None
    label = None
    ps1_path = "data/ps1_g_aligned.npy"
    if os.path.exists(ps1_path):
        raw = np.load(ps1_path)                           # 400x400, +-50"
        real_disp = obs.suppress_stars(raw)               # star-removed nebula
        # downsample to the fit grid (100x100 at 1"/pix) for the geometry fit
        k = real_disp.shape[0] // 100
        real_fit = real_disp[:100 * k, :100 * k].reshape(100, k, 100, k).mean(axis=(1, 3))
        label = "Pan-STARRS1 g (local)"
        print(f"\nUsing saved real image: {ps1_path}  [{label}]")
    else:
        print("\nNo local PS1 image; attempting SkyView download...")
        dl, half, meta = obs.get_real_image(half_arcsec=half, per_survey_timeout_s=25)
        print(f"  coords: {meta['coord_source']}; {meta['message']}")
        if dl is not None:
            real_disp = dl
            real_fit = dl
            label = meta["survey"]

    if real_disp is not None:
        synth, _ = synthetic_image(pm, pixel_scale=2 * half / real_disp.shape[0],
                                   fov_arcsec=2 * half)
        p = obs.save_comparison(real_disp, synth, half,
                                f"{out}/phase4_real_vs_synthetic.png",
                                title=f"NGC 6905: real ({label}) vs synthetic")
        print(f"[saved] real vs synthetic + difference -> {p}")
        print("\nFitting geometry to the REAL image "
              "(broadband/thin-shell mismatch -> modest correlation expected)...")
        rres = fitting.fit_image(real_fit, half, base_config=config, verbose=True)
        import json
        with open(f"{out}/phase4_realfit_params.json", "w") as f:
            json.dump({"best": rres["best"], "start": rres["start"],
                       "best_loss": rres["best_loss"], "real_image": label}, f, indent=2)
        print(f"[saved] real-image best-fit -> {out}/phase4_realfit_params.json")
    else:
        print("  -> No real image available. "
              "Supply a FITS via observations.load_fits_image(path).")

    # 3) Fitting self-consistency demo (always runs; no network needed).
    print("\nFit self-consistency demo (inject known params, recover from image):")
    sres, true_params, target, dhalf = fitting.self_consistency_demo(config)
    print(f"  {'param':24s}{'true':>8s}{'best':>8s}{'err':>8s}")
    for n in fitting.PARAM_NAMES:
        print(f"  {n:24s}{sres['true'][n]:8.2f}{sres['best'][n]:8.2f}"
              f"{sres['best'][n]-sres['true'][n]:+8.2f}")
    print(f"  image match: loss {sres['start_loss']:.4f} -> {sres['best_loss']:.4f} "
          f"(corr {1-sres['best_loss']:.4f})")
    # Demonstrate the inclination-length degeneracy explicitly.
    import numpy as _np
    proj_true = true_params[3] * _np.cos(_np.deg2rad(true_params[0]))
    proj_best = sres['best']['lobe_length_arcsec'] * _np.cos(_np.deg2rad(sres['best']['inclination_deg']))
    print(f"  projected lobe length (length*cos i): true={proj_true:.1f}″, "
          f"best={proj_best:.1f}″  -> same projection (image degeneracy)")
    p = _plot_fit_demo(sres, true_params, target, dhalf, f"{out}/phase4_fit_demo.png")
    print(f"[saved] fit demo figure -> {p}")
    print("\nNOTE: image fitting constrains GEOMETRY only; it cannot constrain")
    print("      expansion velocities or kinematic ages (no velocity info in an image).")
    print("\nPhase 4 complete.")
    return pm


def run_phase5(config=None):
    """Phase 5: interactive browser editor (headless self-test + launch hint)."""
    import editor
    editor._selftest()
    print("\nPhase 5 editor is ready. Launch the browser app with:")
    print("    python editor.py")
    print("then open http://127.0.0.1:8050")
    print("\nPhase 5 complete.")


def run_phase6(config=None):
    """Phase 6: kinematic ages, Monte Carlo uncertainties, ejection history."""
    import ages
    print("=" * 70)
    print("PHASE 6: kinematic ages and evolutionary history")
    print("=" * 70)
    ages.run(config)
    print("\nPhase 6 complete.")


def run_phase7(config=None):
    """Phase 7: interactive 3D HTML, rotation GIF, expansion time-lapse GIF."""
    import viz3d
    print("=" * 70)
    print("PHASE 7: 3D visualization and animations")
    print("=" * 70)
    viz3d.run(config)
    print("\nPhase 7 complete.")


def run_phase8(config=None):
    """Phase 8: export the 3D model to .glb / .obj / .stl and verify load-back."""
    import export3d
    print("=" * 70)
    print("PHASE 8: 3D model export")
    print("=" * 70)
    export3d.run(config)
    print("\nPhase 8 complete.")


def run_all(config=None):
    """Run the full non-interactive pipeline end to end."""
    for name in ["phase1", "phase2", "phase3", "phase4", "phase5",
                 "phase6", "phase7", "phase8"]:
        PHASES[name](config)


PHASES = {"phase1": run_phase1, "phase2": run_phase2, "phase3": run_phase3,
          "phase4": run_phase4, "phase5": run_phase5, "phase6": run_phase6,
          "phase7": run_phase7, "phase8": run_phase8, "all": run_all}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    if which not in PHASES:
        print(f"Unknown phase '{which}'. Available: {list(PHASES)}")
        sys.exit(1)
    PHASES[which]()
