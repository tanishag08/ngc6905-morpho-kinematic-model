"""
editor.py -- Phase 5: interactive browser editor (Plotly Dash), SHAPE-style.

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
from modifiers import ModifierStack, Twist, Bump
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

    ("knot_pos_arcsec", "knot position (arcsec)", 25, 55, 0.5, _d.knot.axial_position_arcsec),
    ("knot_sigma_arcsec", "knot size sigma (arcsec)", 1, 10, 0.5, _d.knot.sigma_arcsec),
    ("knot_vel_kms", "knot v (km/s)", 20, 140, 1, _d.knot.velocity_kms),
    ("knot_emiss", "knot emissivity", 0, 5, 0.1, _d.knot.emissivity),
]
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

    # Optional lobe modifiers (twist about axis y=1, azimuthal bump about y).
    mods = {}
    stack = []
    if params["lobe_twist"] > 0:
        stack.append(Twist(1, params["lobe_twist"]))
    if params["lobe_bump_amp"] > 0:
        stack.append(Bump(1, params["lobe_bump_amp"], frequency=6))
    if stack:
        mods[ID_LOBE] = ModifierStack(stack)
    return conf, mods


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


def _image_heatmap(img, half, title, cmap="Inferno"):
    off = np.linspace(-half, half, img.shape[1])
    fig = go.Figure(go.Heatmap(z=img, x=off, y=off, colorscale=cmap,
                               showscale=False))
    # north up (y normal), east left (x reversed)
    fig.update_xaxes(title="E (arcsec)", autorange="reversed",
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="N (arcsec)")
    fig.update_layout(title=title, margin=dict(l=40, r=10, t=40, b=40),
                      height=330)
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
    fig.update_layout(title=title, margin=dict(l=45, r=10, t=40, b=40), height=330)
    return fig


def fig_3d(pmodel, max_points=2500):
    fig = go.Figure()
    for cid, ps in pmodel["components"].items():
        n = len(ps)
        idx = np.random.default_rng(0).choice(n, size=min(max_points, n), replace=False)
        p = ps.positions[idx]
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=1.6, color=_COMP_COLOR[cid]), name=_COMP_NAME[cid]))
    # line-of-sight arrow (+z away from observer)
    fig.update_layout(scene=dict(
        xaxis_title="x=E (arcsec)", yaxis_title="y=N (arcsec)",
        zaxis_title="z=LOS (arcsec)", aspectmode="cube"),
        margin=dict(l=0, r=0, t=30, b=0), height=430, title="3D particle view")
    return fig


def age_table_html(params):
    ages = kinematic_ages(params)
    rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v:,.0f} yr</td></tr>"
        for k, v in ages.items())
    spread = max(ages.values()) - min(ages.values())
    return (f"<table style='width:100%;font-family:monospace'>"
            f"<tr><th>component</th><th style='text-align:right'>kinematic age</th></tr>"
            f"{rows}</table>"
            f"<div style='font-family:monospace;margin-top:4px'>"
            f"max−min = {spread:,.0f} yr</div>")


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
def build_app():
    from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

    app = Dash(__name__)
    half = _d.grid.half_size_arcsec

    def slider_row(sid, label, lo, hi, step, default):
        tag = "  [published]" if sid in PUBLISHED else ""
        return html.Div([
            html.Label(label + tag, style={"fontSize": "12px"}),
            dcc.Slider(lo, hi, step, value=default, id=sid,
                       marks=None, tooltip={"placement": "bottom"},
                       updatemode="mouseup"),
        ], style={"marginBottom": "6px"})

    controls = [slider_row(*s) for s in SLIDERS]
    buttons = html.Div([
        html.Button("Render (full)", id="render_btn", n_clicks=0),
        html.Button("Reset", id="reset_btn", n_clicks=0, style={"marginLeft": "6px"}),
        html.Button("Save params", id="save_btn", n_clicks=0, style={"marginLeft": "6px"}),
        dcc.Download(id="download_params"),
        dcc.Upload(html.Button("Load params", style={"marginLeft": "6px"}),
                   id="upload_params"),
    ], style={"margin": "8px 0"})

    app.layout = html.Div([
        html.H3("NGC 6905 — SHAPE-style interactive editor"),
        html.Div([
            html.Div([buttons] + controls,
                     style={"width": "25%", "float": "left", "maxHeight": "88vh",
                            "overflowY": "scroll", "paddingRight": "10px"}),
            html.Div([
                html.Div([
                    html.Div(dcc.Graph(id="g_synth"), style={"width": "50%", "display": "inline-block"}),
                    html.Div(dcc.Graph(id="g_real"), style={"width": "50%", "display": "inline-block"}),
                ]),
                html.Div([
                    html.Div(dcc.Graph(id="g_pv_major"), style={"width": "50%", "display": "inline-block"}),
                    html.Div(dcc.Graph(id="g_pv_minor"), style={"width": "50%", "display": "inline-block"}),
                ]),
                html.Div([
                    html.Div(dcc.Graph(id="g_3d"), style={"width": "60%", "display": "inline-block"}),
                    html.Div([html.H4("Kinematic ages"),
                              html.Div(id="age_box")],
                             style={"width": "38%", "display": "inline-block",
                                    "verticalAlign": "top"}),
                ]),
                html.H4("Velocity channel maps (updated on full render)"),
                html.Img(id="channel_img", style={"width": "100%"}),
            ], style={"width": "74%", "float": "right"}),
        ]),
    ], style={"fontFamily": "sans-serif", "padding": "10px"})

    # --- main update: sliders (live, low-N) or render button (full + channels) ---
    @app.callback(
        [Output("g_synth", "figure"), Output("g_real", "figure"),
         Output("g_pv_major", "figure"), Output("g_pv_minor", "figure"),
         Output("g_3d", "figure"), Output("age_box", "children"),
         Output("channel_img", "src")],
        [Input(sid, "value") for sid in SLIDER_IDS] + [Input("render_btn", "n_clicks")],
    )
    def update(*vals):
        params = dict(zip(SLIDER_IDS, [float(v) for v in vals[:-1]]))
        full = (ctx.triggered_id == "render_btn")
        pmodel = compute_model(params, N_FULL if full else N_LIVE)
        figs = (fig_synthetic(pmodel, half), fig_real(half),
                fig_pv(pmodel, cfg.DEFAULT.render.pv_slit_pa_major_deg, "P-V major (PA 160°)"),
                fig_pv(pmodel, cfg.DEFAULT.render.pv_slit_pa_minor_deg, "P-V minor (PA 70°)"),
                fig_3d(pmodel), html.Iframe(srcDoc=age_table_html(params),
                                            style={"border": 0, "width": "100%", "height": "140px"}))
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
    f4 = fig_3d(pm)
    ages = kinematic_ages(params)
    assert all(np.isfinite(list(ages.values())))
    # modifiers path
    params2 = dict(params, lobe_twist=4.0, lobe_bump_amp=3.0)
    pm2 = compute_model(params2, 2000)
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
