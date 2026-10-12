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
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import config as cfg

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # = 1/2.3548

# ---------------------------------------------------------------------------
# Particle rendering (Phase 3). These mirror the grid renderers above but take
# a particle-model dict (from particles.build_particle_model) and bin the
# particle cloud directly. Same conventions: north up, east left; v_los = v_z.
# ---------------------------------------------------------------------------


def particle_sky_image(pmodel, n_bins=None):
    """Bin a particle cloud into a sky image (sum of emissivity per pixel).

    Returns (image, half) with image shape (ny, nx), row=north(y), col=east(x).
    """
    conf = pmodel["config"]
    half = conf.grid.half_size_arcsec
    if n_bins is None:
        n_bins = conf.grid.n
    pts = pmodel["particles"].positions
    w = pmodel["particles"].weights
    edges = np.linspace(-half, half, n_bins + 1)
    # histogram2d with x=east along axis0, y=north along axis1, then transpose
    # so rows=north, cols=east (matching the grid sky_image convention).
    H2, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges, edges], weights=w)
    return H2.T, half


def particle_pv_diagram(pmodel, pa_deg, slit_width_arcsec=None, n_pos_bins=140):
    """Long-slit P-V diagram from a particle cloud.

    Returns (pv, pos_edges, vel_edges), pv shape (n_vel, n_pos).
    """
    conf = pmodel["config"]
    if slit_width_arcsec is None:
        slit_width_arcsec = conf.render.pv_slit_width_arcsec
    ps = pmodel["particles"]
    x, y = ps.positions[:, 0], ps.positions[:, 1]
    pos, perp = _slit_coordinates(x, y, pa_deg)
    in_slit = np.abs(perp) <= slit_width_arcsec / 2.0

    half = conf.grid.half_size_arcsec
    pos_edges = np.linspace(-half, half, n_pos_bins + 1)
    vel_edges = np.arange(conf.render.pv_v_min_kms,
                          conf.render.pv_v_max_kms + conf.render.pv_v_bin_kms,
                          conf.render.pv_v_bin_kms)
    pv, _, _ = np.histogram2d(ps.v_los[in_slit], pos[in_slit],
                              bins=[vel_edges, pos_edges], weights=ps.weights[in_slit])
    return pv, pos_edges, vel_edges


def plot_particle_sky_image(pmodel, save_path, title="Particle sky image"):
    image, half = particle_sky_image(pmodel)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(image, origin="lower", extent=[-half, half, -half, half], cmap="inferno")
    ax.invert_xaxis()
    ax.set_xlabel("east offset (arcsec)  [east is left]")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title(title)
    ax.text(0.02, 0.98, f"i = {pmodel['inclination_deg']:.0f}°, "
            f"PA = {pmodel['position_angle_deg']:.0f}°", transform=ax.transAxes,
            va="top", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="integrated emissivity (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_particle_pv(pmodel, pa_deg, save_path, title=None, slit_width_arcsec=None):
    pv, pos_edges, vel_edges = particle_pv_diagram(pmodel, pa_deg, slit_width_arcsec)
    if title is None:
        title = f"Particle P-V diagram, slit PA = {pa_deg:.0f}°"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(pv, origin="lower", aspect="auto", cmap="inferno",
                   extent=[pos_edges[0], pos_edges[-1], vel_edges[0], vel_edges[-1]])
    ax.set_xlabel("position along slit (arcsec)")
    ax.set_ylabel("line-of-sight velocity (km/s)\n(+ = redshift/receding)")
    ax.axhline(0, color="cyan", lw=0.7, alpha=0.6)
    ax.axvline(0, color="cyan", lw=0.7, alpha=0.6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="emissivity (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


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


# ===========================================================================
# Phase 4: realistic synthetic observations
# ===========================================================================

# Colour per component for composite views.
COMPONENT_COLORS = {cfg.ID_SHELL: (0.2, 0.7, 1.0),   # cyan   - shell
                    cfg.ID_LOBE: (1.0, 0.6, 0.1),    # orange - lobes
                    cfg.ID_KNOT: (1.0, 0.2, 0.2)}    # red    - knots


def _particle_subset(pmodel, component_id):
    """Return the ParticleSet for one component, or the combined set."""
    if component_id is None:
        return pmodel["particles"]
    return pmodel["components"][component_id]


def synthetic_image(pmodel, pixel_scale=None, seeing_fwhm=None, fov_arcsec=None,
                    component_id=None):
    """Synthetic sky image at a given pixel scale, convolved with Gaussian seeing.

    Returns (image, half) with image shape (ny, nx), row=north, col=east.
    Seeing is applied as a Gaussian blur of sigma = FWHM / 2.3548 (in pixels).
    """
    conf = pmodel["config"]
    pixel_scale = conf.render.pixel_scale_arcsec if pixel_scale is None else pixel_scale
    seeing_fwhm = conf.render.seeing_fwhm_arcsec if seeing_fwhm is None else seeing_fwhm
    fov_arcsec = 2 * conf.grid.half_size_arcsec if fov_arcsec is None else fov_arcsec
    half = fov_arcsec / 2.0
    nbin = max(1, int(round(fov_arcsec / pixel_scale)))
    edges = np.linspace(-half, half, nbin + 1)

    ps = _particle_subset(pmodel, component_id)
    H2, _, _ = np.histogram2d(ps.positions[:, 0], ps.positions[:, 1],
                              bins=[edges, edges], weights=ps.weights)
    img = H2.T
    if seeing_fwhm and seeing_fwhm > 0:
        img = gaussian_filter(img, seeing_fwhm * FWHM_TO_SIGMA / pixel_scale)
    return img, half


def plot_synthetic_image(pmodel, save_path, pixel_scale=None, seeing_fwhm=None,
                         title="Synthetic image (with seeing)"):
    conf = pmodel["config"]
    pixel_scale = conf.render.pixel_scale_arcsec if pixel_scale is None else pixel_scale
    seeing_fwhm = conf.render.seeing_fwhm_arcsec if seeing_fwhm is None else seeing_fwhm
    img, half = synthetic_image(pmodel, pixel_scale, seeing_fwhm)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(img, origin="lower", extent=[-half, half, -half, half], cmap="inferno")
    ax.invert_xaxis()
    ax.set_xlabel("east offset (arcsec)  [east is left]")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title(f"{title}\npixel {pixel_scale}″, seeing {seeing_fwhm}″ FWHM")
    fig.colorbar(im, ax=ax, label="surface brightness (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path


def composite_image(pmodel, pixel_scale=None, seeing_fwhm=None, fov_arcsec=None):
    """RGB composite with each component in its own colour."""
    conf = pmodel["config"]
    pixel_scale = conf.render.pixel_scale_arcsec if pixel_scale is None else pixel_scale
    rgb = None
    half = None
    for cid in pmodel["components"]:
        img, half = synthetic_image(pmodel, pixel_scale, seeing_fwhm, fov_arcsec,
                                    component_id=cid)
        if img.max() > 0:
            img = img / img.max()                  # normalise each component
        color = np.array(COMPONENT_COLORS[cid])
        layer = img[:, :, None] * color[None, None, :]
        rgb = layer if rgb is None else rgb + layer
    rgb = np.clip(rgb, 0, 1)
    return rgb, half


def plot_composite_image(pmodel, save_path, pixel_scale=None, seeing_fwhm=None,
                         title="Composite (shell=cyan, lobes=orange, knots=red)"):
    rgb, half = composite_image(pmodel, pixel_scale, seeing_fwhm)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half])
    ax.invert_xaxis()
    ax.set_xlabel("east offset (arcsec)  [east is left]")
    ax.set_ylabel("north offset (arcsec)")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path


def synthetic_pv(pmodel, pa_deg, slit_width_arcsec=None, slit_offset_arcsec=0.0,
                 instrumental_fwhm_kms=None, seeing_fwhm_arcsec=None,
                 n_pos_bins=140, component_id=None):
    """P-V diagram with a configurable slit (PA, width, offset) and smoothing.

    The slit centre line is offset perpendicular to its length by
    slit_offset_arcsec. Emission is Gaussian-smoothed along velocity by the
    instrumental FWHM and (optionally) along position by the seeing FWHM.
    Returns (pv, pos_edges, vel_edges).
    """
    conf = pmodel["config"]
    if slit_width_arcsec is None:
        slit_width_arcsec = conf.render.pv_slit_width_arcsec
    if instrumental_fwhm_kms is None:
        instrumental_fwhm_kms = conf.render.pv_instrumental_fwhm_kms

    ps = _particle_subset(pmodel, component_id)
    x, y = ps.positions[:, 0], ps.positions[:, 1]
    pos, perp = _slit_coordinates(x, y, pa_deg)
    in_slit = np.abs(perp - slit_offset_arcsec) <= slit_width_arcsec / 2.0

    half = conf.grid.half_size_arcsec
    pos_edges = np.linspace(-half, half, n_pos_bins + 1)
    vel_edges = np.arange(conf.render.pv_v_min_kms,
                          conf.render.pv_v_max_kms + conf.render.pv_v_bin_kms,
                          conf.render.pv_v_bin_kms)
    pv, _, _ = np.histogram2d(ps.v_los[in_slit], pos[in_slit],
                              bins=[vel_edges, pos_edges], weights=ps.weights[in_slit])

    # Instrumental smoothing along velocity (axis 0).
    if instrumental_fwhm_kms and instrumental_fwhm_kms > 0:
        sigma_v = instrumental_fwhm_kms * FWHM_TO_SIGMA / conf.render.pv_v_bin_kms
        pv = gaussian_filter1d(pv, sigma_v, axis=0)
    # Optional seeing smoothing along position (axis 1).
    if seeing_fwhm_arcsec and seeing_fwhm_arcsec > 0:
        pos_bin = (pos_edges[1] - pos_edges[0])
        sigma_p = seeing_fwhm_arcsec * FWHM_TO_SIGMA / pos_bin
        pv = gaussian_filter1d(pv, sigma_p, axis=1)
    return pv, pos_edges, vel_edges


def plot_synthetic_pv(pmodel, pa_deg, save_path, title=None, **kwargs):
    pv, pos_edges, vel_edges = synthetic_pv(pmodel, pa_deg, **kwargs)
    if title is None:
        title = f"Synthetic P-V (PA {pa_deg:.0f}°, instrumentally smoothed)"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(pv, origin="lower", aspect="auto", cmap="inferno",
                   extent=[pos_edges[0], pos_edges[-1], vel_edges[0], vel_edges[-1]])
    ax.set_xlabel("position along slit (arcsec)")
    ax.set_ylabel("line-of-sight velocity (km/s)\n(+ = redshift)")
    ax.axhline(0, color="cyan", lw=0.7, alpha=0.6)
    ax.axvline(0, color="cyan", lw=0.7, alpha=0.6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="emissivity (arb.)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path


def channel_maps(pmodel, v_min=None, v_max=None, channel_width=None,
                 pixel_scale=None, seeing_fwhm=None, fov_arcsec=None):
    """Slice the particle cloud into velocity channels and image each.

    Returns (images, v_edges, half): images is a list of 2-D arrays, one per
    channel; v_edges are the channel boundaries (km/s).
    """
    conf = pmodel["config"]
    v_min = conf.render.channel_v_min_kms if v_min is None else v_min
    v_max = conf.render.channel_v_max_kms if v_max is None else v_max
    channel_width = conf.render.channel_width_kms if channel_width is None else channel_width
    pixel_scale = conf.render.pixel_scale_arcsec if pixel_scale is None else pixel_scale
    seeing_fwhm = conf.render.seeing_fwhm_arcsec if seeing_fwhm is None else seeing_fwhm
    fov_arcsec = 2 * conf.grid.half_size_arcsec if fov_arcsec is None else fov_arcsec

    half = fov_arcsec / 2.0
    nbin = max(1, int(round(fov_arcsec / pixel_scale)))
    edges = np.linspace(-half, half, nbin + 1)
    v_edges = np.arange(v_min, v_max + channel_width, channel_width)

    ps = pmodel["particles"]
    images = []
    for a, b in zip(v_edges[:-1], v_edges[1:]):
        m = (ps.v_los >= a) & (ps.v_los < b)
        H2, _, _ = np.histogram2d(ps.positions[m, 0], ps.positions[m, 1],
                                  bins=[edges, edges], weights=ps.weights[m])
        img = H2.T
        if seeing_fwhm and seeing_fwhm > 0:
            img = gaussian_filter(img, seeing_fwhm * FWHM_TO_SIGMA / pixel_scale)
        images.append(img)
    return images, v_edges, half


def plot_channel_maps(pmodel, save_path, title="Velocity channel maps", **kwargs):
    images, v_edges, half = channel_maps(pmodel, **kwargs)
    n = len(images)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    vmax = max((im.max() for im in images if im.max() > 0), default=1.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow),
                             squeeze=False)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_xticks([]); ax.set_yticks([])
        if idx < n:
            ax.imshow(images[idx], origin="lower",
                      extent=[-half, half, -half, half], cmap="inferno",
                      vmin=0, vmax=vmax)
            ax.invert_xaxis()
            ax.text(0.04, 0.9, f"[{v_edges[idx]:.0f},{v_edges[idx+1]:.0f}] km/s",
                    transform=ax.transAxes, color="white", fontsize=7.5)
        else:
            ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path
