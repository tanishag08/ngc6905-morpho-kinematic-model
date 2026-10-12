"""
editor.py -- SHAPE: interactive morpho-kinematic modeller for NGC 6905 (Dash).

A browser mock of the SHAPE software (Steffen & Lopez) workflow: modules
(3D / Physics / Graphs / Rendering / Optimizer), an object tree with a live
modifier stack (Twist / Wiggle-Bump / Size-curve / Bend), mesh & particle
viewports, synthetic-vs-observation rendering, P-V diagrams, channel maps, and
a kinematic-age readout, on a fully black SHAPE-style canvas.

Launch with:
    python editor.py          # then open http://127.0.0.1:8050 in a browser

Layout:
  * Left: sliders for every parameter -- global (inclination, PA, distance),
    shell (size, thickness, v_exp, emissivity), lobes (length, opening, wall,
    tip velocity, emissivity, twist + bump modifiers), knots (position, size,
    velocity, emissivity).
  * Right: live panels that update together --
      - 3D particle view
      - synthetic sky image  next to  the real Pan-STARRS image
      - P-V diagrams for both slits (major PA 160, minor PA 70)
      - velocity channel maps (on full render)
      - a live kinematic-age readout (shell / lobes / knots)
  * Buttons: Render (full resolution), Save params (JSON), Load params (JSON
    upload), Reset to published starting values.

Responsiveness: slider drags recompute with a reduced particle count; the
"Render (full)" button recomputes at full resolution and adds channel maps.

All the compute/figure functions are plain functions (no Dash) so they can be
unit-tested headlessly; `_selftest()` exercises them without a browser.
"""

import io
import json
import base64
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg
from config import ID_SHELL, ID_LOBE, ID_KNOT
from particles import build_particle_model
from modifiers import ModifierStack, Twist, Bump, SizeCurve, Bend
from render import (synthetic_image, synthetic_pv, channel_maps,
                    plot_channel_maps)
from units import kinematic_age_from_arcsec

# Particle counts: small while dragging, large on explicit full render.
N_LIVE = 8000
N_FULL = 120000

REAL_IMAGE_PATH = "data/ps1_g_aligned.npy"   # aligned Pan-STARRS g cutout (Phase 4)

# ---------------------------------------------------------------------------
# Slider definitions: (id, label, min, max, step, default)
# Defaults are taken from the published/assumption config so "Reset" is exact.
# ---------------------------------------------------------------------------
_d = cfg.DEFAULT
SLIDERS = [
    ("inclination_deg", "inclination i (deg)", 0, 90, 1, _d.geometry.inclination_deg),
    ("position_angle_deg", "position angle PA (deg)", 100, 200, 1, _d.geometry.position_angle_deg),
    ("distance_kpc", "distance (kpc)", 2.0, 3.4, 0.05, _d.geometry.distance_kpc),

    ("shell_radius_arcsec", "shell radius (arcsec)", 10, 30, 0.5, _d.shell.semi_axes_arcsec[0]),
    ("shell_thickness_arcsec", "shell thickness (arcsec)", 1, 10, 0.5, _d.shell.thickness_arcsec),
    ("shell_vexp_kms", "shell v_exp (km/s)", 20, 80, 1, _d.shell.expansion_velocity_kms),
    ("shell_emiss", "shell emissivity", 0, 3, 0.1, _d.shell.emissivity),

    ("lobe_length_arcsec", "lobe length (arcsec)", 25, 55, 0.5, _d.lobe.length_arcsec),
    ("lobe_opening_deg", "lobe half-opening (deg)", 8, 35, 0.5, _d.lobe.half_opening_angle_deg),
    ("lobe_wall_arcsec", "lobe wall (arcsec)", 1, 10, 0.5, _d.lobe.wall_thickness_arcsec),
    ("lobe_vtip_kms", "lobe tip v (km/s)", 20, 120, 1, _d.lobe.tip_velocity_kms),
    ("lobe_emiss", "lobe emissivity", 0, 3, 0.1, _d.lobe.emissivity),
    ("lobe_twist", "lobe twist (deg/arcsec) [mod]", 0, 10, 0.5, 0.0),
    ("lobe_bump_amp", "lobe bump amplitude (arcsec) [mod]", 0, 6, 0.5, 0.0),
    ("lobe_bend_deg", "lobe bend (deg) [mod]", -60, 60, 1, 0.0),

    ("knot_pos_arcsec", "knot position (arcsec)", 25, 55, 0.5, _d.knot.axial_position_arcsec),
    ("knot_sigma_arcsec", "knot size sigma (arcsec)", 1, 10, 0.5, _d.knot.sigma_arcsec),
    ("knot_vel_kms", "knot v (km/s)", 20, 140, 1, _d.knot.velocity_kms),
    ("knot_emiss", "knot emissivity", 0, 5, 0.1, _d.knot.emissivity),

    # SHAPE 'Size' modifier for the lobes: 5 curve control points (base -> tip),
    # each a radius scale factor. All = 1.0 means the curve is flat (no change).
    ("lobe_size_0", "Size curve · base", 0.2, 2.0, 0.05, 1.0),
    ("lobe_size_1", "Size curve · 1", 0.2, 2.0, 0.05, 1.0),
    ("lobe_size_2", "Size curve · 2", 0.2, 2.0, 0.05, 1.0),
    ("lobe_size_3", "Size curve · 3", 0.2, 2.0, 0.05, 1.0),
    ("lobe_size_4", "Size curve · tip", 0.2, 2.0, 0.05, 1.0),
]
LOBE_SIZE_IDS = [f"lobe_size_{i}" for i in range(5)]
SLIDER_IDS = [s[0] for s in SLIDERS]
DEFAULT_PARAMS = {s[0]: s[5] for s in SLIDERS}

# Which parameters are published (for display / "moved from published" hints).
PUBLISHED = {"inclination_deg", "position_angle_deg", "distance_kpc",
             "shell_radius_arcsec", "shell_vexp_kms", "lobe_length_arcsec"}


# ---------------------------------------------------------------------------
# Params -> config -> particle model
# ---------------------------------------------------------------------------
def make_config(params):
    """Build a Config + orientation + per-component modifiers from slider params."""
    r = params["shell_radius_arcsec"]
    shell = replace(_d.shell, semi_axes_arcsec=(r, r, r),
                    thickness_arcsec=params["shell_thickness_arcsec"],
                    expansion_velocity_kms=params["shell_vexp_kms"],
                    v_char_radius_arcsec=r, emissivity=params["shell_emiss"])
    L = params["lobe_length_arcsec"]
    lobe = replace(_d.lobe, length_arcsec=L,
                   half_opening_angle_deg=params["lobe_opening_deg"],
                   wall_thickness_arcsec=params["lobe_wall_arcsec"],
                   tip_velocity_kms=params["lobe_vtip_kms"],
                   v_char_radius_arcsec=L, emissivity=params["lobe_emiss"])
    knot = replace(_d.knot, axial_position_arcsec=params["knot_pos_arcsec"],
                   sigma_arcsec=params["knot_sigma_arcsec"],
                   velocity_kms=params["knot_vel_kms"],
                   v_char_radius_arcsec=params["knot_pos_arcsec"],
                   emissivity=params["knot_emiss"])
    geometry = replace(_d.geometry, distance_kpc=params["distance_kpc"],
                       inclination_deg=params["inclination_deg"],
                       position_angle_deg=params["position_angle_deg"])
    conf = replace(_d, shell=shell, lobe=lobe, knot=knot, geometry=geometry)

    # Optional lobe modifier stack (applied in order), SHAPE-style.
    mods = {}
    stack = []
    if params["lobe_twist"] > 0:
        stack.append(Twist(1, params["lobe_twist"]))
    if params["lobe_bump_amp"] > 0:
        stack.append(Bump(1, params["lobe_bump_amp"], frequency=6))
    size_pts = [params[i] for i in LOBE_SIZE_IDS]
    if any(abs(s - 1.0) > 1e-6 for s in size_pts):       # Size curve not flat
        stack.append(SizeCurve(1, size_pts))
    if abs(params.get("lobe_bend_deg", 0.0)) > 1e-6:
        stack.append(Bend(1, params["lobe_bend_deg"]))
    if stack:
        mods[ID_LOBE] = ModifierStack(stack)
    return conf, mods


def active_modifiers(params):
    """Return the list of active lobe modifier names (for the stack display)."""
    names = []
    if params["lobe_twist"] > 0:
        names.append(f"Twist  ({params['lobe_twist']:.1f}°/″)")
    if params["lobe_bump_amp"] > 0:
        names.append(f"Wiggle/Bump  (amp {params['lobe_bump_amp']:.1f}″)")
    if any(abs(params[i] - 1.0) > 1e-6 for i in LOBE_SIZE_IDS):
        names.append("Size  (curve)")
    if abs(params.get("lobe_bend_deg", 0.0)) > 1e-6:
        names.append(f"Bend  ({params['lobe_bend_deg']:.0f}°)")
    return names


def compute_model(params, n_particles):
    conf, mods = make_config(params)
    return build_particle_model(conf, inclination_deg=params["inclination_deg"],
                                position_angle_deg=params["position_angle_deg"],
                                n_per_component=n_particles, modifiers=mods,
                                match_grid_flux=False)


def kinematic_ages(params):
    """Return dict of per-component kinematic ages (yr).

    Age = r_char(true 3D) / v. The slider sets the TRUE 3D characteristic radius,
    so NO inclination deprojection is applied here (inclination only changes the
    projected appearance, not the intrinsic size). Deprojection matters only when
    sizes are MEASURED from a projected image (Phase 6).
    """
    d_pc = params["distance_kpc"] * 1000.0
    return {
        "shell": float(kinematic_age_from_arcsec(params["shell_radius_arcsec"],
                                                 params["shell_vexp_kms"], d_pc)),
        "lobes": float(kinematic_age_from_arcsec(params["lobe_length_arcsec"],
                                                 params["lobe_vtip_kms"], d_pc)),
        "knots": float(kinematic_age_from_arcsec(params["knot_pos_arcsec"],
                                                 params["knot_vel_kms"], d_pc)),
    }


# ---------------------------------------------------------------------------
# Figure builders (plotly, browser-interactive)
# ---------------------------------------------------------------------------
_COMP_COLOR = {ID_SHELL: "#33b5ff", ID_LOBE: "#ff9a1a", ID_KNOT: "#ff3333"}
_COMP_NAME = {ID_SHELL: "shell", ID_LOBE: "lobes", ID_KNOT: "knots"}

# SHAPE look: fully black viewports/windows (as SHAPE renders), light text.
PANEL_BG = "#000000"
PAPER_BG = "#000000"
BORDER = "#2b2b2b"
ACCENT = "#56c5ff"
FG = "#e6e6e6"
_DARK = dict(template="plotly_dark", paper_bgcolor=PAPER_BG, plot_bgcolor=PANEL_BG,
             font=dict(color=FG, family="Consolas, monospace", size=11))


def _image_heatmap(img, half, title, cmap="Inferno"):
    off = np.linspace(-half, half, img.shape[1])
    fig = go.Figure(go.Heatmap(z=img, x=off, y=off, colorscale=cmap,
                               showscale=False))
    # north up (y normal), east left (x reversed)
    fig.update_xaxes(title="E (arcsec)", autorange="reversed",
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="N (arcsec)")
    fig.update_layout(title=title, margin=dict(l=40, r=10, t=36, b=40),
                      height=320, **_DARK)
    return fig


def fig_synthetic(pmodel, half):
    img, _ = synthetic_image(pmodel, pixel_scale=0.5, seeing_fwhm=1.0,
                             fov_arcsec=2 * half)
    return _image_heatmap(img, half, "Synthetic image (model)")


def fig_real(half):
    img = load_real_image(half)
    if img is None:
        fig = go.Figure()
        fig.add_annotation(text="No real image found<br>(data/ps1_g_aligned.npy)",
                           showarrow=False)
        fig.update_layout(title="Real image", height=330)
        return fig
    return _image_heatmap(img, half, "Real image (Pan-STARRS g)")


def fig_pv(pmodel, pa_deg, title):
    pv, pe, ve = synthetic_pv(pmodel, pa_deg)
    fig = go.Figure(go.Heatmap(z=pv, x=np.linspace(pe[0], pe[-1], pv.shape[1]),
                               y=np.linspace(ve[0], ve[-1], pv.shape[0]),
                               colorscale="Inferno", showscale=False))
    fig.update_xaxes(title="position (arcsec)")
    fig.update_yaxes(title="v_los (km/s)")
    fig.update_layout(title=title, margin=dict(l=45, r=10, t=36, b=40),
                      height=320, **_DARK)
    return fig


def _oriented_component_meshes(params):
    """Component meshes (shell/lobes/knots) rotated body->sky for the viewport."""
    from meshes import component_meshes
    from orientation import body_to_sky
    conf, _ = make_config(params)
    raw = component_meshes(conf)
    groups = {ID_SHELL: ["shell"], ID_LOBE: ["lobe_N", "lobe_S"],
              ID_KNOT: ["knot_N", "knot_S"]}
    out = {}
    i, pa = params["inclination_deg"], params["position_angle_deg"]
    for cid, names in groups.items():
        verts, faces, offset = [], [], 0
        for nm in names:
            m = raw[nm]
            v = body_to_sky(np.asarray(m.vertices), i, pa)
            f = np.asarray(m.faces) + offset
            verts.append(v); faces.append(f); offset += len(v)
        out[cid] = (np.vstack(verts), np.vstack(faces))
    return out


def fig_3d(pmodel, params, mode="Particles (component)", max_points=2500):
    """SHAPE-style model viewport: particles and/or mesh surfaces."""
    fig = go.Figure()
    half = pmodel["config"].grid.half_size_arcsec
    rng = np.random.default_rng(0)

    show_particles = mode != "Mesh"
    show_mesh = mode in ("Mesh", "Mesh + particles")
    color_vlos = mode == "Particles (v_los)"

    if show_particles and not color_vlos:
        for cid, ps in pmodel["components"].items():
            n = len(ps); idx = rng.choice(n, size=min(max_points, n), replace=False)
            p = ps.positions[idx]
            fig.add_trace(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=1.6, color=_COMP_COLOR[cid]), name=_COMP_NAME[cid]))
    if color_vlos:
        allp = pmodel["particles"].positions; allv = pmodel["particles"].v_los
        idx = rng.choice(len(allp), size=min(max_points * 2, len(allp)), replace=False)
        vmax = float(np.percentile(np.abs(allv), 98)) or 1.0
        fig.add_trace(go.Scatter3d(
            x=allp[idx, 0], y=allp[idx, 1], z=allp[idx, 2], mode="markers",
            marker=dict(size=1.6, color=allv[idx],
                        colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
                        cmin=-vmax, cmax=vmax, colorbar=dict(title="v_los")),
            name="v_los"))
    if show_mesh:
        meshes = _oriented_component_meshes(params)
        for cid, (v, f) in meshes.items():
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=_COMP_COLOR[cid], opacity=0.35, flatshading=True,
                name=_COMP_NAME[cid]))

    # line-of-sight marker (+z away from observer)
    fig.add_trace(go.Scatter3d(x=[0.8 * half, 0.8 * half], y=[0.8 * half, 0.8 * half],
                               z=[-half, half], mode="lines",
                               line=dict(color=ACCENT, width=4), name="line of sight"))
    fig.update_layout(scene=dict(
        xaxis_title="x=E", yaxis_title="y=N", zaxis_title="z=LOS",
        aspectmode="cube", bgcolor=PANEL_BG),
        margin=dict(l=0, r=0, t=28, b=0), height=460, title="MODEL viewport",
        legend=dict(x=0, y=1), **_DARK)
    return fig


def age_table_html(params):
    ages = kinematic_ages(params)
    swatch = {"shell": "#33b5ff", "lobes": "#ff9a1a", "knots": "#ff3333"}
    rows = "".join(
        f"<tr><td><span style='color:{swatch[k]}'>&#9632;</span> {k}</td>"
        f"<td style='text-align:right'>{v:,.0f} yr</td></tr>"
        for k, v in ages.items())
    spread = max(ages.values()) - min(ages.values())
    return (f"<html><body style='margin:0;background:#000000;color:#e6e6e6;"
            f"font-family:Consolas,monospace;font-size:12px'>"
            f"<table style='width:100%'>"
            f"<tr style='color:#9aa4b2'><th style='text-align:left'>component</th>"
            f"<th style='text-align:right'>kinematic age</th></tr>{rows}</table>"
            f"<div style='margin-top:6px;color:#9aa4b2'>max−min = {spread:,.0f} yr</div>"
            f"</body></html>")


def fig_size_curve(params):
    """SHAPE 'Graphs' module: the lobe Size modifier curve (radius scale vs axis)."""
    y = [params[i] for i in LOBE_SIZE_IDS]
    x = np.linspace(0, 1, len(y))
    xf = np.linspace(0, 1, 100)
    yf = np.interp(xf, x, y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines", line=dict(color=ACCENT, width=2),
                             name="Size curve"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(size=9, color="#ff9a1a"), name="control pts"))
    fig.add_hline(y=1.0, line=dict(color="#555", dash="dot"))
    fig.update_xaxes(title="along lobe axis   base → tip", range=[-0.02, 1.02])
    fig.update_yaxes(title="radius scale", range=[0, 2.1])
    fig.update_layout(title="Size modifier curve (Graphs)", height=260,
                      margin=dict(l=45, r=10, t=36, b=40), showlegend=False, **_DARK)
    return fig


def modstack_html(params):
    """SHAPE-style modifier stack readout for each object."""
    mods = active_modifiers(params)
    lobe_mods = ("<br>".join(f"&nbsp;&nbsp;▸ {m}" for m in mods)
                 if mods else "&nbsp;&nbsp;(none)")
    return (f"<html><body style='margin:0;background:#000000;color:#e6e6e6;"
            f"font-family:Consolas,monospace;font-size:11px'>"
            f"<div style='color:#9aa4b2'>MODIFIER STACK</div>"
            f"<div><span style='color:#33b5ff'>■</span> shell: &nbsp;(base ellipsoid)</div>"
            f"<div><span style='color:#ff9a1a'>■</span> lobes:</div>{lobe_mods}"
            f"<div><span style='color:#ff3333'>■</span> knots: &nbsp;(base Gaussians)</div>"
            f"</body></html>")


def channel_maps_png_b64(pmodel):
    """Render channel maps to a base64 PNG (for the full-render panel)."""
    buf = io.BytesIO()
    import matplotlib
    matplotlib.use("Agg")
    path = "/tmp/_editor_channels.png"
    plot_channel_maps(pmodel, path, title="Velocity channel maps")
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Real image loader (with light star suppression + asinh stretch for display)
# ---------------------------------------------------------------------------
def load_real_image(half=50.0):
    import os
    if not os.path.exists(REAL_IMAGE_PATH):
        return None
    img = np.load(REAL_IMAGE_PATH)
    from observations import suppress_stars
    neb = suppress_stars(img)
    lo, hi = np.nanpercentile(neb, [40, 99.3])
    return np.arcsinh(np.clip((neb - lo) / (hi - lo + 1e-9), 0, 1) * 10)


def real_fit_target(half=50.0):
    """Star-suppressed observation downsampled to the fit grid (100x100), or None."""
    import os
    if not os.path.exists(REAL_IMAGE_PATH):
        return None, half
    from observations import suppress_stars
    neb = suppress_stars(np.load(REAL_IMAGE_PATH))
    k = neb.shape[0] // 100
    target = neb[:100 * k, :100 * k].reshape(100, k, 100, k).mean(axis=(1, 3))
    return target, half


# ---------------------------------------------------------------------------
# Params save / load helpers
# ---------------------------------------------------------------------------
def params_to_json(params):
    return json.dumps(params, indent=2)


def params_from_json(text):
    data = json.loads(text)
    # keep only known keys, fall back to defaults for any missing
    return {k: float(data.get(k, DEFAULT_PARAMS[k])) for k in SLIDER_IDS}


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
# Module organisation (mirrors SHAPE's Structure / Physics / Render modules).
SLIDER_BY_ID = {s[0]: s for s in SLIDERS}
STRUCTURE_SECTIONS = [
    ("Orientation / scene", ["inclination_deg", "position_angle_deg", "distance_kpc"]),
    ("Shell  (ellipsoid)", ["shell_radius_arcsec", "shell_thickness_arcsec"]),
    ("Lobes  (bipolar cones)", ["lobe_length_arcsec", "lobe_opening_deg", "lobe_wall_arcsec"]),
    ("Lobe modifiers  (twist / bump / bend)", ["lobe_twist", "lobe_bump_amp", "lobe_bend_deg"]),
    ("Knots  (ansae)", ["knot_pos_arcsec", "knot_sigma_arcsec"]),
]
PHYSICS_SECTIONS = [
    ("Density  (emissivity)", ["shell_emiss", "lobe_emiss", "knot_emiss"]),
    ("Velocity  (homologous v = H·r)", ["shell_vexp_kms", "lobe_vtip_kms", "knot_vel_kms"]),
]

# Fully-black SHAPE styles.
S_APP = {"backgroundColor": "#000000", "color": FG, "fontFamily": "Consolas, monospace",
         "padding": "8px", "minHeight": "100vh"}
S_PANEL = {"backgroundColor": "#000000", "border": "1px solid " + BORDER,
           "borderRadius": "3px", "padding": "8px", "margin": "4px"}
S_BTN = {"backgroundColor": "#111111", "color": FG, "border": "1px solid #3a3a3a",
         "borderRadius": "3px", "padding": "6px 10px", "marginRight": "6px",
         "cursor": "pointer", "fontFamily": "Consolas, monospace"}
S_TAB = {"backgroundColor": "#000000", "color": "#8a8a8a", "border": "1px solid " + BORDER,
         "padding": "6px"}
S_TAB_SEL = {"backgroundColor": "#0a0a0a", "color": ACCENT, "border": "1px solid " + BORDER,
             "borderTop": "2px solid " + ACCENT, "padding": "6px"}


def build_app():
    from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

    app = Dash(__name__)
    app.title = "SHAPE — NGC 6905"
    half = _d.grid.half_size_arcsec

    def slider_row(sid):
        _, label, lo, hi, step, default = SLIDER_BY_ID[sid]
        tag = " ★" if sid in PUBLISHED else ""     # star = published value
        return html.Div([
            html.Label(label + tag, style={"fontSize": "11px", "color": "#b8c0cc"}),
            dcc.Slider(lo, hi, step, value=default, id=sid, marks=None,
                       tooltip={"placement": "bottom", "always_visible": False},
                       updatemode="mouseup"),
        ], style={"marginBottom": "2px"})

    def section(title, ids):
        return html.Div([html.Div(title, style={"color": ACCENT, "fontWeight": "bold",
                                                 "fontSize": "12px", "margin": "6px 0 2px"})]
                        + [slider_row(i) for i in ids])

    def module(sections):
        return html.Div([section(t, ids) for t, ids in sections],
                        style={"maxHeight": "74vh", "overflowY": "auto", "paddingRight": "6px"})

    # Object tree + live modifier stack (SHAPE's object/modifier hierarchy).
    tree = html.Div([
        html.Iframe(id="modstack", style={"border": 0, "width": "100%",
                                          "height": "120px", "backgroundColor": PAPER_BG}),
    ], style=dict(S_PANEL, **{"fontSize": "12px"}))

    # Render module controls.
    render_module = html.Div([
        html.Div("Viewport mode", style={"color": ACCENT, "fontSize": "12px", "fontWeight": "bold"}),
        dcc.RadioItems(id="view_mode",
                       options=["Particles (component)", "Particles (v_los)",
                                "Mesh", "Mesh + particles"],
                       value="Particles (component)",
                       style={"fontSize": "12px"},
                       labelStyle={"display": "block", "margin": "2px 0"}),
        html.Hr(style={"borderColor": "#2a2f37"}),
        html.Button("► RENDER (full res + channel maps)", id="render_btn", n_clicks=0,
                    style=dict(S_BTN, **{"width": "100%", "marginBottom": "6px"})),
        html.Div("Dragging sliders uses a reduced particle count for speed; "
                 "RENDER does the full pass.", style={"fontSize": "11px", "color": "#8a93a0"}),
    ])

    toolbar = html.Div([
        html.Span("SHAPE", style={"fontWeight": "bold", "color": ACCENT,
                                  "fontSize": "18px", "letterSpacing": "2px"}),
        html.Span("  NGC 6905  ", style={"color": FG, "fontSize": "14px"}),
        html.Span("[model]   ", style={"color": "#777"}),
        html.Button("Reset ★published", id="reset_btn", n_clicks=0, style=S_BTN),
        html.Button("Save", id="save_btn", n_clicks=0, style=S_BTN),
        dcc.Upload(html.Button("Load", style=S_BTN), id="upload_params",
                   style={"display": "inline-block"}),
        dcc.Download(id="download_params"),
    ], style=dict(S_PANEL, **{"marginBottom": "6px"}))

    # Graphs module: the SHAPE 'Size' modifier curve (control points + plot).
    graphs_module = html.Div([
        section("Lobe Size modifier  (SHAPE 'Size')", LOBE_SIZE_IDS),
        dcc.Graph(id="g_sizecurve", config={"displayModeBar": False}),
        html.Div("Set the control points to shape the lobe radius from base to "
                 "tip (1.0 = unchanged). This is SHAPE's curve-driven Size modifier.",
                 style={"fontSize": "11px", "color": "#8a93a0"}),
    ], style={"maxHeight": "74vh", "overflowY": "auto", "paddingRight": "6px"})

    # Optimizer module: fit the model geometry to the observation.
    optimizer_module = html.Div([
        html.Div("Optimizer — fit model to observation", style={"color": ACCENT,
                 "fontSize": "12px", "fontWeight": "bold", "margin": "6px 0"}),
        html.Button("► FIT GEOMETRY TO OBSERVATION", id="optimize_btn", n_clicks=0,
                    style=dict(S_BTN, **{"width": "100%", "marginBottom": "6px"})),
        html.Iframe(id="optimize_status", style={"border": 0, "width": "100%",
                                                 "height": "130px", "backgroundColor": "#000"}),
        html.Div("Optimises inclination, PA, shell radius, lobe length and opening "
                 "angle to best match the observation image (Powell). Geometry only "
                 "— velocities are not constrained by an image.",
                 style={"fontSize": "11px", "color": "#8a8a8a"}),
    ], style={"maxHeight": "74vh", "overflowY": "auto", "paddingRight": "6px"})

    tabs = dcc.Tabs([
        dcc.Tab(module(STRUCTURE_SECTIONS), label="3D",
                style=S_TAB, selected_style=S_TAB_SEL),
        dcc.Tab(module(PHYSICS_SECTIONS), label="Physics",
                style=S_TAB, selected_style=S_TAB_SEL),
        dcc.Tab(graphs_module, label="Graphs",
                style=S_TAB, selected_style=S_TAB_SEL),
        dcc.Tab(render_module, label="Rendering",
                style=S_TAB, selected_style=S_TAB_SEL),
        dcc.Tab(optimizer_module, label="Optimizer",
                style=S_TAB, selected_style=S_TAB_SEL),
    ], style={"height": "34px"})

    def viewport(title, graph_id, w="50%"):
        return html.Div(dcc.Graph(id=graph_id, config={"displModeBar": False}),
                        style={"width": w, "display": "inline-block",
                               "verticalAlign": "top"})

    left = html.Div([tree, html.Div(tabs, style=S_PANEL)],
                    style={"width": "26%", "float": "left"})
    right = html.Div([
        html.Div([  # MODEL viewport + ages
            html.Div(dcc.Graph(id="g_3d", config={"displayModeBar": False}),
                     style=dict(S_PANEL, **{"width": "62%", "display": "inline-block"})),
            html.Div([html.Div("KINEMATIC AGES", style={"color": ACCENT, "fontSize": "12px"}),
                      html.Iframe(id="age_box", style={"border": 0, "width": "100%",
                                                       "height": "150px",
                                                       "backgroundColor": PAPER_BG})],
                     style=dict(S_PANEL, **{"width": "34%", "display": "inline-block",
                                            "verticalAlign": "top"})),
        ]),
        html.Div([  # RENDER module: synthetic vs observation
            html.Div("RENDER — model vs observation",
                     style={"color": ACCENT, "fontSize": "12px", "marginLeft": "6px"}),
            viewport("synthetic", "g_synth"), viewport("observation", "g_real"),
        ], style=S_PANEL),
        html.Div([  # P-V module
            html.Div("POSITION–VELOCITY (long-slit)",
                     style={"color": ACCENT, "fontSize": "12px", "marginLeft": "6px"}),
            viewport("pv major", "g_pv_major"), viewport("pv minor", "g_pv_minor"),
        ], style=S_PANEL),
        html.Div([  # channel maps
            html.Div("VELOCITY CHANNEL MAPS (RENDER to update)",
                     style={"color": ACCENT, "fontSize": "12px"}),
            html.Img(id="channel_img", style={"width": "100%"}),
        ], style=S_PANEL),
    ], style={"width": "73%", "float": "right"})

    app.layout = html.Div([toolbar, left, right], style=S_APP)

    # --- main update: sliders/view (live, low-N) or RENDER (full + channels) ---
    @app.callback(
        [Output("g_synth", "figure"), Output("g_real", "figure"),
         Output("g_pv_major", "figure"), Output("g_pv_minor", "figure"),
         Output("g_3d", "figure"), Output("age_box", "srcDoc"),
         Output("g_sizecurve", "figure"), Output("modstack", "srcDoc"),
         Output("channel_img", "src")],
        [Input(sid, "value") for sid in SLIDER_IDS]
        + [Input("render_btn", "n_clicks"), Input("view_mode", "value")],
    )
    def update(*vals):
        params = dict(zip(SLIDER_IDS, [float(v) for v in vals[:len(SLIDER_IDS)]]))
        view_mode = vals[-1]
        full = (ctx.triggered_id == "render_btn")
        pmodel = compute_model(params, N_FULL if full else N_LIVE)
        figs = (fig_synthetic(pmodel, half), fig_real(half),
                fig_pv(pmodel, cfg.DEFAULT.render.pv_slit_pa_major_deg, "P-V major (PA 160°)"),
                fig_pv(pmodel, cfg.DEFAULT.render.pv_slit_pa_minor_deg, "P-V minor (PA 70°)"),
                fig_3d(pmodel, params, mode=view_mode), age_table_html(params),
                fig_size_curve(params), modstack_html(params))
        channels = channel_maps_png_b64(pmodel) if full else no_update
        return (*figs, channels)

    # --- save params ---
    @app.callback(Output("download_params", "data"), Input("save_btn", "n_clicks"),
                  [State(sid, "value") for sid in SLIDER_IDS], prevent_initial_call=True)
    def save(n, *vals):
        params = dict(zip(SLIDER_IDS, [float(v) for v in vals]))
        return dict(content=params_to_json(params), filename="ngc6905_params.json")

    # --- reset to published defaults ---
    @app.callback([Output(sid, "value") for sid in SLIDER_IDS],
                  Input("reset_btn", "n_clicks"), prevent_initial_call=True)
    def reset(n):
        return [DEFAULT_PARAMS[sid] for sid in SLIDER_IDS]

    # --- load params from uploaded JSON ---
    @app.callback([Output(sid, "value", allow_duplicate=True) for sid in SLIDER_IDS],
                  Input("upload_params", "contents"), prevent_initial_call=True)
    def load(contents):
        if not contents:
            return [no_update] * len(SLIDER_IDS)
        _, b64 = contents.split(",", 1)
        text = base64.b64decode(b64).decode()
        p = params_from_json(text)
        return [p[sid] for sid in SLIDER_IDS]

    # --- Optimizer: fit geometry to the observation, push best-fit to sliders ---
    import fitting
    FIT_IDS = fitting.PARAM_NAMES   # inclination, PA, shell radius, lobe length, opening

    @app.callback(
        [Output(sid, "value", allow_duplicate=True) for sid in FIT_IDS]
        + [Output("optimize_status", "srcDoc")],
        Input("optimize_btn", "n_clicks"),
        [State(sid, "value") for sid in SLIDER_IDS], prevent_initial_call=True)
    def optimize(n, *vals):
        params = dict(zip(SLIDER_IDS, [float(v) for v in vals]))
        target, hlf = real_fit_target(half)
        if target is None:
            return [no_update] * len(FIT_IDS) + [
                "<body style='color:#e6e6e6;background:#000;font-family:monospace;"
                "font-size:11px'>No observation image found.</body>"]
        start = [params[k] for k in FIT_IDS]
        res = fitting.fit_image(target, hlf, base_config=make_config(params)[0],
                                start_params=start, verbose=False)
        best = [res["best"][k] for k in FIT_IDS]
        rows = "".join(
            f"<tr><td>{k.replace('_',' ')}</td><td style='text-align:right'>"
            f"{res['start'][k]:.1f}</td><td style='text-align:right;color:#56c5ff'>"
            f"{res['best'][k]:.1f}</td></tr>" for k in FIT_IDS)
        html_status = (
            f"<body style='margin:0;color:#e6e6e6;background:#000;font-family:monospace;"
            f"font-size:11px'><div style='color:#56c5ff'>FIT corr = "
            f"{1 - res['best_loss']:.3f}</div>"
            f"<table style='width:100%'><tr style='color:#888'><td>param</td>"
            f"<td style='text-align:right'>start</td><td style='text-align:right'>"
            f"best</td></tr>{rows}</table></body>")
        return best + [html_status]

    return app


# ---------------------------------------------------------------------------
# Headless self-test (no browser needed)
# ---------------------------------------------------------------------------
def _selftest():
    print("editor self-test:")
    params = dict(DEFAULT_PARAMS)
    pm = compute_model(params, 4000)
    half = _d.grid.half_size_arcsec
    f1 = fig_synthetic(pm, half)
    f2 = fig_real(half)
    f3 = fig_pv(pm, 160, "major")
    # exercise every viewport mode (particles / v_los / mesh / both)
    for mode in ["Particles (component)", "Particles (v_los)", "Mesh", "Mesh + particles"]:
        _ = fig_3d(pm, params, mode=mode)
    ages = kinematic_ages(params)
    assert all(np.isfinite(list(ages.values())))
    # modifiers path incl. SHAPE Size-curve + Bend
    params2 = dict(params, lobe_twist=4.0, lobe_bump_amp=3.0,
                   lobe_size_0=1.4, lobe_size_4=0.4, lobe_bend_deg=25.0)
    pm2 = compute_model(params2, 2000)
    assert len(active_modifiers(params2)) == 4   # twist, bump, size, bend
    _ = fig_size_curve(params2); _ = modstack_html(params2)
    # optimizer target available?
    tgt, _h = real_fit_target(half)
    print(f"  optimizer target present: {tgt is not None}")
    # save/load round-trip
    txt = params_to_json(params)
    back = params_from_json(txt)
    assert all(abs(back[k] - params[k]) < 1e-9 for k in SLIDER_IDS)
    print(f"  figures built OK; ages={ {k: round(v) for k, v in ages.items()} }")
    print(f"  real image present: {load_real_image(half) is not None}")
    print(f"  modifier path OK ({len(pm2['particles'])} particles)")
    print(f"  save/load round-trip OK ({len(SLIDER_IDS)} params)")
    print("  app constructs:", end=" ")
    app = build_app()
    print(type(app).__name__, "OK")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print("Starting NGC 6905 editor at http://127.0.0.1:8050  (Ctrl-C to stop)")
        build_app().run(debug=False, port=8050)
