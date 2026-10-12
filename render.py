"""
render.py -- Turn a 3D model into observables: sky images and P-V diagrams.

Phase 1 implements:
  * sky_image      -- emissivity summed along the line of sight (z axis)
  * pv_diagram     -- a long-slit position-velocity diagram

Display convention (see orientation.py): north UP, east LEFT. We achieve east-
left by inverting the x (east) axis after plotting with a normal extent.

Later phases (4) add seeing, pixel scale, channel maps, and instrumental
smoothing; those parameters already live in config.RenderParams.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config as cfg


# --- Sky image -------------------------------------------------------------

def sky_image(model):
    """Sum the combined emissivity cube along the line of sight (z, axis 2).

    Returns (image, half) where image has shape (ny, nx) with row=north(y),
    col=east(x), suitable for imshow(origin='lower').
    """
    emiss = model["emissivity"]              # (nx, ny, nz)
    img_xy = emiss.sum(axis=2)               # integrate over z (LOS) -> (nx, ny)
    # Transpose so rows correspond to y (north) and columns to x (east).
    image = img_xy.T                         # (ny, nx)
    half = model["config"].grid.half_size_arcsec
    return image, half


def plot_sky_image(model, save_path, title="NGC 6905 synthetic sky image"):
    """Plot and save the synthetic sky image (north up, east left)."""
    image, half = sky_image(model)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(image, origin="lower", extent=[-half, half, -half, half],
                   cmap="inferno")
    ax.invert_xaxis()                        # east to the LEFT
    ax.set_xlabel("east offset (arcsec)  [east is left]")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title(title)
    # Annotate orientation.
    ax.text(0.02, 0.98, f"i = {model['inclination_deg']:.0f}°, "
                        f"PA = {model['position_angle_deg']:.0f}°",
            transform=ax.transAxes, va="top", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="integrated emissivity (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


# --- Position-velocity diagram --------------------------------------------

def _slit_coordinates(X, Y, pa_deg):
    """Return (pos, perp) sky-plane coordinates relative to a slit at PA.

    pos  = distance along the slit (arcsec), measured from the star.
    perp = perpendicular distance from the slit centre line (arcsec).

    Slit direction unit vector (from north through east) in (x=east, y=north):
        u = (sin PA, cos PA)        (PA=0 -> north, PA=90 -> east)
    Perpendicular:
        w = (cos PA, -sin PA)
    """
    pa = np.deg2rad(pa_deg)
    pos = X * np.sin(pa) + Y * np.cos(pa)
    perp = X * np.cos(pa) - Y * np.sin(pa)
    return pos, perp


def pv_diagram(model, pa_deg, slit_width_arcsec=None, n_pos_bins=140):
    """Build a long-slit position-velocity histogram.

    Returns (pv, pos_edges, vel_edges) where pv has shape
    (n_vel_bins, n_pos_bins): rows = velocity, cols = slit position.

    For each component the line-of-sight velocity is v_los = H * z (homologous),
    weighted by the component's emissivity. Only voxels within +/- width/2 of the
    slit centre line contribute.
    """
    conf = model["config"]
    if slit_width_arcsec is None:
        slit_width_arcsec = conf.render.pv_slit_width_arcsec

    X, Y, Z = model["X"], model["Y"], model["Z"]
    pos, perp = _slit_coordinates(X, Y, pa_deg)
    in_slit = np.abs(perp) <= slit_width_arcsec / 2.0

    half = conf.grid.half_size_arcsec
    pos_edges = np.linspace(-half, half, n_pos_bins + 1)
    vel_edges = np.arange(conf.render.pv_v_min_kms,
                          conf.render.pv_v_max_kms + conf.render.pv_v_bin_kms,
                          conf.render.pv_v_bin_kms)

    pv = np.zeros((len(vel_edges) - 1, len(pos_edges) - 1))

    pos_slit = pos[in_slit]
    z_slit = Z[in_slit]
    for comp in model["components"].values():
        e = comp["emissivity"][in_slit]
        mask = e > 0
        if not np.any(mask):
            continue
        v_los = comp["H"] * z_slit[mask]     # homologous: v_los = H * z
        hist, _, _ = np.histogram2d(
            v_los, pos_slit[mask],
            bins=[vel_edges, pos_edges],
            weights=e[mask],
        )
        pv += hist
    return pv, pos_edges, vel_edges


def plot_pv(model, pa_deg, save_path, title=None, slit_width_arcsec=None):
    """Plot and save a P-V diagram for a slit at the given PA."""
    pv, pos_edges, vel_edges = pv_diagram(model, pa_deg, slit_width_arcsec)
    if title is None:
        title = f"P-V diagram, slit PA = {pa_deg:.0f}°"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        pv, origin="lower", aspect="auto", cmap="inferno",
        extent=[pos_edges[0], pos_edges[-1], vel_edges[0], vel_edges[-1]],
    )
    # Slit position increases from north-through-east along the slit; for the
    # major-axis (PA 160) slit, positive position is the SE direction.
    ax.set_xlabel("position along slit (arcsec)")
    ax.set_ylabel("line-of-sight velocity (km/s)\n(+ = redshift/receding)")
    ax.axhline(0, color="cyan", lw=0.7, alpha=0.6)   # systemic velocity (subtracted)
    ax.axvline(0, color="cyan", lw=0.7, alpha=0.6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="emissivity (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
