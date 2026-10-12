"""
viz3d.py -- Phase 7: interactive 3D visualization and animations.

Produces:
  * interactive_3d_html : a standalone Plotly HTML with each component a
    different colour and buttons to toggle
      - particles coloured by component
      - particles coloured by line-of-sight velocity (blue approaching / red
        receding)
      - mesh surfaces
      - velocity arrows
    plus a line-of-sight arrow (+z = away = redshift) and an AU scale bar.
  * rotation_gif        : a 360-degree spin of the particle cloud.
  * expansion_timelapse_gif : homologous growth of the nebula from t=0 to
    1.5x the current age, with EACH COMPONENT starting at its own ejection time
    (ballistic / constant-velocity assumption, stated on every frame).

The animations are rendered with matplotlib (3D scatter) and stitched into GIFs
with imageio, so no browser/orca/kaleido is required.
"""

import os

import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import config as cfg
from config import ID_SHELL, ID_LOBE, ID_KNOT
from particles import build_particle_model
from meshes import component_meshes
from orientation import body_to_sky
from units import arcsec_to_au
import ages as ages_mod

COMPONENT_COLORS = {ID_SHELL: "#33b5ff", ID_LOBE: "#ff9a1a", ID_KNOT: "#ff3333"}
COMPONENT_NAMES = {ID_SHELL: "shell", ID_LOBE: "lobes", ID_KNOT: "knots"}

# Blue (approaching, -v) -> white -> red (receding, +v): matches our sign
# convention (z away from observer = redshift = positive v_los).
VLOS_COLORSCALE = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]


# ---------------------------------------------------------------------------
# Interactive Plotly HTML
# ---------------------------------------------------------------------------
def interactive_3d_html(pmodel, save_path, config=None, max_points=4000,
                        n_arrows=150):
    """Build a standalone interactive 3D HTML view with toggle buttons."""
    if config is None:
        config = pmodel.get("config", cfg.DEFAULT)
    rng = np.random.default_rng(config.seed)
    half = config.grid.half_size_arcsec
    d_pc = config.geometry.distance_kpc * 1000.0

    traces = []
    group = []   # tag each trace so buttons can show/hide groups

    # (a) particles coloured by component
    for cid, ps in pmodel["components"].items():
        n = len(ps); idx = rng.choice(n, size=min(max_points, n), replace=False)
        p = ps.positions[idx]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=1.7, color=COMPONENT_COLORS[cid]),
            name=COMPONENT_NAMES[cid], visible=True))
        group.append("comp")

    # (b) particles coloured by line-of-sight velocity (single trace)
    allp = pmodel["particles"].positions
    allv = pmodel["particles"].v_los
    idx = rng.choice(len(allp), size=min(max_points * 2, len(allp)), replace=False)
    vmax = float(np.percentile(np.abs(allv), 98)) or 1.0
    traces.append(go.Scatter3d(
        x=allp[idx, 0], y=allp[idx, 1], z=allp[idx, 2], mode="markers",
        marker=dict(size=1.7, color=allv[idx], colorscale=VLOS_COLORSCALE,
                    cmin=-vmax, cmax=vmax,
                    colorbar=dict(title="v_los (km/s)", x=1.02)),
        name="v_los", visible=False))
    group.append("vlos")

    # (c) mesh surfaces (oriented body -> sky)
    meshes = component_meshes(config)
    mesh_cid = {"shell": ID_SHELL, "lobe_N": ID_LOBE, "lobe_S": ID_LOBE,
                "knot_N": ID_KNOT, "knot_S": ID_KNOT}
    for name, m in meshes.items():
        v = body_to_sky(np.asarray(m.vertices), config.geometry.inclination_deg,
                        config.geometry.position_angle_deg)
        f = np.asarray(m.faces)
        cid = mesh_cid[name]
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=COMPONENT_COLORS[cid], opacity=0.35, name=name,
            visible=False, showscale=False))
        group.append("mesh")

    # (d) velocity arrows (cones), subsampled
    ps = pmodel["particles"]
    ai = rng.choice(len(ps.positions), size=min(n_arrows, len(ps.positions)), replace=False)
    ap = ps.positions[ai]; av = ps.velocities[ai]
    traces.append(go.Cone(
        x=ap[:, 0], y=ap[:, 1], z=ap[:, 2], u=av[:, 0], v=av[:, 1], w=av[:, 2],
        sizemode="absolute", sizeref=8, showscale=False, colorscale="Greys",
        name="velocity arrows", visible=False))
    group.append("arrows")

    # (e) always-on: line-of-sight arrow (+z away from observer) and AU scale bar
    xo, yo = 0.85 * half, 0.85 * half
    traces.append(go.Scatter3d(
        x=[xo, xo], y=[yo, yo], z=[-half, half], mode="lines",
        line=dict(color="black", width=4), name="line of sight", visible=True,
        hoverinfo="name"))
    group.append("always")
    traces.append(go.Cone(
        x=[xo], y=[yo], z=[half], u=[0], v=[0], w=[1], sizemode="absolute",
        sizeref=10, showscale=False, colorscale=[[0, "black"], [1, "black"]],
        visible=True, name="+z (away, redshift)"))
    group.append("always")
    # scale bar: a 20-arcsec segment near the bottom, labelled in AU
    bar = 20.0
    au = arcsec_to_au(bar, d_pc)
    traces.append(go.Scatter3d(
        x=[-0.9 * half, -0.9 * half + bar], y=[-0.9 * half, -0.9 * half],
        z=[-0.95 * half, -0.95 * half], mode="lines+text",
        line=dict(color="white", width=6),
        text=["", f"{bar:.0f}″ = {au:,.0f} AU"], textposition="top center",
        name="scale bar", visible=True))
    group.append("always")

    # Visibility helpers for the buttons.
    def vis(show):
        return [g in show or g == "always" for g in group]

    buttons = [
        dict(label="Particles: by component", method="update",
             args=[{"visible": vis({"comp"})}]),
        dict(label="Particles: by v_los", method="update",
             args=[{"visible": vis({"vlos"})}]),
        dict(label="Mesh surfaces", method="update",
             args=[{"visible": vis({"mesh"})}]),
        dict(label="Velocity arrows + components", method="update",
             args=[{"visible": vis({"comp", "arrows"})}]),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="NGC 6905 — interactive 3D model "
              f"(i={config.geometry.inclination_deg:.0f}°, "
              f"PA={config.geometry.position_angle_deg:.0f}°)",
        scene=dict(xaxis_title="x = East (arcsec)", yaxis_title="y = North (arcsec)",
                   zaxis_title="z = line of sight (arcsec)", aspectmode="cube"),
        updatemenus=[dict(type="buttons", direction="down", x=0.0, y=1.0,
                          buttons=buttons)],
        margin=dict(l=0, r=0, t=40, b=0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    return save_path


# ---------------------------------------------------------------------------
# matplotlib frame helpers for GIFs
# ---------------------------------------------------------------------------
def _scatter_frame(groups, lim, elev, azim, title, subtitle=None):
    """Render one 3D scatter frame; groups = list of (positions, color)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for pos, color in groups:
        if len(pos):
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=2, c=color,
                       alpha=0.5, edgecolors="none")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("E"); ax.set_ylabel("N"); ax.set_zlabel("LOS")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    if subtitle:
        ax.text2D(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8,
                  color="dimgray")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def _subsample(pmodel, per_component, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for cid, ps in pmodel["components"].items():
        n = len(ps); idx = rng.choice(n, size=min(per_component, n), replace=False)
        groups[cid] = ps.positions[idx]
    return groups


def rotation_gif(pmodel, save_path, n_frames=36, per_component=2500, lim=None):
    """360-degree rotation of the particle cloud, saved as a GIF."""
    if lim is None:
        lim = pmodel["config"].grid.half_size_arcsec
    groups = _subsample(pmodel, per_component)
    frames = []
    for k in range(n_frames):
        azim = 360.0 * k / n_frames
        g = [(groups[cid], COMPONENT_COLORS[cid]) for cid in groups]
        frames.append(_scatter_frame(g, lim, elev=18, azim=azim,
                                     title="NGC 6905 — 3D model (rotating)",
                                     subtitle="colours: shell/lobes/knots"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.09, loop=0)
    return save_path


def expansion_timelapse_gif(config=None, save_path="outputs/phase7_expansion.gif",
                            n_frames=40, per_component=2500):
    """Homologous growth from t=0 to 1.5x current age; each component starts at
    its own ejection time (constant-velocity / ballistic assumption).

    At absolute time tau (years since the FIRST ejection), a component with
    current kinematic age t_c has been expanding for (tau - (t_oldest - t_c)),
    so its size scales by that divided by t_c (=1 at tau = t_oldest = 'now').
    """
    if config is None:
        config = cfg.DEFAULT
    # Current kinematic ages (nominal, deprojected) from Phase 6.
    rows = ages_mod.age_table(config)
    age = {r["component"]: r["age_yr"] for r in rows}
    name_to_cid = {"shell": ID_SHELL, "lobes": ID_LOBE, "knots": ID_KNOT}
    t_c = {name_to_cid[k]: v for k, v in age.items()}
    t_oldest = max(t_c.values())

    # Build the current model once; scale positions per frame.
    pmodel = build_particle_model(config, n_per_component=per_component * 3,
                                  match_grid_flux=False)
    groups_now = _subsample(pmodel, per_component)
    lim = config.grid.half_size_arcsec * 1.5

    frames = []
    for k in range(n_frames):
        tau = 1.5 * t_oldest * k / (n_frames - 1)
        g = []
        for cid, pos in groups_now.items():
            expand_time = tau - (t_oldest - t_c[cid])    # years this comp has expanded
            scale = max(0.0, expand_time / t_c[cid])     # =1 at 'now'
            if scale <= 0:
                continue                                 # not yet ejected
            g.append((pos * scale, COMPONENT_COLORS[cid]))
        now_tag = "  <-- NOW" if abs(tau - t_oldest) < (1.5 * t_oldest / n_frames) else ""
        frames.append(_scatter_frame(
            g, lim, elev=18, azim=35,
            title=f"NGC 6905 homologous expansion\n{tau:,.0f} yr since first ejection{now_tag}",
            subtitle="ASSUMES constant velocities; each component starts at its own ejection time"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.12, loop=0)
    return save_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(config=None, outputs_dir=None, n_particles=60000):
    if config is None:
        config = cfg.DEFAULT
    out = outputs_dir or config.outputs_dir
    pmodel = build_particle_model(config, n_per_component=n_particles,
                                  match_grid_flux=False)
    p = interactive_3d_html(pmodel, f"{out}/phase7_interactive_3d.html", config)
    print(f"[saved] interactive 3D HTML -> {p}")
    p = rotation_gif(pmodel, f"{out}/phase7_rotation.gif")
    print(f"[saved] rotation GIF -> {p}")
    p = expansion_timelapse_gif(config, f"{out}/phase7_expansion.gif")
    print(f"[saved] expansion time-lapse GIF -> {p}")
    return pmodel


if __name__ == "__main__":
    run()
