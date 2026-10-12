"""
observations.py -- Download and align a real image of NGC 6905.

Strategy (per the project spec):
  1. Resolve the central-star coordinates from SIMBAD.
  2. Try to download an image: HST (MAST) first, then DSS / Pan-STARRS via
     SkyView (astroquery).
  3. Align to the model frame: centred on the star, calibrated to arcsec
     offsets, displayed north-up / east-left.

Everything network-dependent is wrapped so that a failure is reported plainly
and the caller can supply a local FITS/image file instead (load_fits_image).

The aligned product is returned as (image, half_arcsec, meta) where `image` is a
2-D array with row=north, col=east on a square field of half-width `half_arcsec`,
matching the model's synthetic images so the two can be compared directly.
"""

import os
import signal
import contextlib
import numpy as np


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Hard wall-clock limit using SIGALRM (main thread only).

    Ensures a slow/hung network call can never stall the pipeline. If SIGALRM is
    unavailable (e.g. not the main thread), it becomes a no-op.
    """
    def _handler(signum, frame):
        raise _Timeout(f"timed out after {seconds}s")
    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(seconds))
    except (ValueError, AttributeError):
        yield
        return
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# NGC 6905 fallback coordinates (J2000) if SIMBAD is unreachable.
# RA 20h22m23.0s, Dec +20d06m16s  (SIMBAD, HD 193949 / central star).
FALLBACK_RA_DEG = 305.5958
FALLBACK_DEC_DEG = 20.1044
TARGET_NAME = "NGC 6905"


def get_target_coordinates(name=TARGET_NAME):
    """Resolve (ra_deg, dec_deg) from SIMBAD; fall back to hard-coded values.

    Returns (ra_deg, dec_deg, source_string).
    """
    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        result = Simbad.query_object(name)
        if result is not None and len(result) > 0:
            # astroquery >=0.4.8 returns columns 'ra'/'dec' in degrees.
            cols = {c.lower(): c for c in result.colnames}
            if "ra" in cols and "dec" in cols:
                ra = float(result[cols["ra"]][0])
                dec = float(result[cols["dec"]][0])
                # Some versions return sexagesimal strings; handle both.
                if not np.isfinite(ra):
                    raise ValueError("non-finite RA")
                return ra, dec, "SIMBAD"
    except Exception as exc:  # network / parsing / version issues
        print(f"  [observations] SIMBAD lookup failed ({exc}); using fallback coords.")
    return FALLBACK_RA_DEG, FALLBACK_DEC_DEG, "fallback (hard-coded)"


def download_skyview(ra_deg, dec_deg, surveys=None, fov_arcmin=3.0, pixels=600,
                     per_survey_timeout_s=30):
    """Try to download a cutout via SkyView for the first survey that works.

    Each survey request is bounded by per_survey_timeout_s so a hung request
    cannot stall the pipeline. Returns (data, wcs, survey_name) or
    (None, None, None) on total failure.
    """
    if surveys is None:
        # Order: deeper/sharper first. DSS2 Red captures Halpha+[N II] reasonably
        # for a bright PN; DSS is the lighter/faster fallback.
        surveys = ["DSS2 Red", "DSS2 IR", "DSS"]
    try:
        from astroquery.skyview import SkyView
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS
        import astropy.units as u
    except Exception as exc:
        print(f"  [observations] astroquery/astropy import failed: {exc}")
        return None, None, None

    pos = SkyCoord(ra_deg, dec_deg, unit="deg")
    for survey in surveys:
        try:
            with time_limit(per_survey_timeout_s):
                imgs = SkyView.get_images(position=pos, survey=[survey],
                                          width=fov_arcmin * u.arcmin,
                                          height=fov_arcmin * u.arcmin,
                                          pixels=str(pixels))
            if imgs:
                hdu = imgs[0][0]
                return np.asarray(hdu.data, dtype=float), WCS(hdu.header), survey
        except _Timeout as exc:
            print(f"  [observations] SkyView '{survey}' {exc}")
            continue
        except Exception as exc:
            print(f"  [observations] SkyView '{survey}' failed: {exc!r}")
            continue
    return None, None, None


def _align_from_wcs(data, wcs, ra_deg, dec_deg, half_arcsec):
    """Resample a WCS image onto a square arcsec grid centred on the star.

    Simple nearest/bilinear sampling via WCS world->pixel; returns a (ny, nx)
    array with row=north (increasing up), col=east (increasing left when later
    displayed with invert_xaxis), to match the model convention.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    n = 400
    off = np.linspace(-half_arcsec, half_arcsec, n)          # arcsec
    east, north = np.meshgrid(off, off)                      # east=+x, north=+y
    # Convert offsets to world coordinates about the centre.
    cosd = np.cos(np.deg2rad(dec_deg))
    ra = ra_deg - (east / 3600.0) / cosd                     # east increases RA? east = -RA direction
    dec = dec_deg + north / 3600.0
    sky = SkyCoord(ra, dec, unit="deg")
    try:
        px, py = wcs.world_to_pixel(sky)
    except Exception:
        xy = wcs.wcs_world2pix(np.column_stack([ra.ravel(), dec.ravel()]), 0)
        px = xy[:, 0].reshape(ra.shape); py = xy[:, 1].reshape(ra.shape)
    px = np.round(px).astype(int); py = np.round(py).astype(int)
    valid = (px >= 0) & (px < data.shape[1]) & (py >= 0) & (py < data.shape[0])
    out = np.full((n, n), np.nan)
    out[valid] = data[py[valid], px[valid]]
    return out


def get_real_image(half_arcsec=50.0, name=TARGET_NAME, per_survey_timeout_s=30):
    """High-level: resolve coords, download, align. Returns (image, half, meta).

    On failure returns (None, half, meta) with meta describing what happened so
    the caller can ask the user for a local file. Each survey download is
    time-bounded so the pipeline never hangs on a slow network.
    """
    ra, dec, src = get_target_coordinates(name)
    meta = {"ra_deg": ra, "dec_deg": dec, "coord_source": src,
            "survey": None, "ok": False}
    data, wcs, survey = download_skyview(ra, dec, pixels=400,
                                         per_survey_timeout_s=per_survey_timeout_s)
    if data is None:
        meta["message"] = ("Image download failed (no network or survey "
                           "unavailable). Supply a local FITS with "
                           "load_fits_image(path).")
        return None, half_arcsec, meta
    aligned = _align_from_wcs(data, wcs, ra, dec, half_arcsec)
    meta.update(survey=survey, ok=True,
                message=f"Downloaded {survey}, aligned about {src} coords.")
    return aligned, half_arcsec, meta


def load_fits_image(path, ra_deg=None, dec_deg=None, half_arcsec=50.0):
    """Load a user-supplied FITS image and align it to the model grid.

    If ra/dec are None, SIMBAD/fallback coordinates are used for the centre.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    if ra_deg is None or dec_deg is None:
        ra_deg, dec_deg, _ = get_target_coordinates()
    with fits.open(path) as hdul:
        hdu = next(h for h in hdul if getattr(h, "data", None) is not None)
        data = np.asarray(hdu.data, dtype=float)
        wcs = WCS(hdu.header)
    aligned = _align_from_wcs(data, wcs, ra_deg, dec_deg, half_arcsec)
    return aligned, half_arcsec, {"survey": os.path.basename(path), "ok": True}


def load_wfpc_waiver(path, ra_deg=None, dec_deg=None, half_arcsec=40.0, out_n=800):
    """Load an HST WFPC/WFPC2 'waiver' FITS (c0f) and mosaic its 4 chips.

    A waiver c0f stores the 4 CCD images as a (4, ny, nx) cube in the primary
    HDU, with per-group WCS (CRVAL/CRPIX/CD) in the table HDU. We build a WCS per
    chip and resample all four onto a square arcsec grid centred on (ra,dec),
    north-up / east-left, filling the characteristic rosette (chips meet near the
    field centre). Returns (mosaic, half, meta).

    NOTE for NGC 6905: the only HST image found was WFPC1 F785LP (red long-pass),
    in which the nebular emission lines are not captured -- so the nebula is
    essentially undetected. This loader is kept for reproducibility / future
    narrowband WFPC2 data.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    if ra_deg is None or dec_deg is None:
        ra_deg, dec_deg, _ = get_target_coordinates()
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=float)      # (ngroup, ny, nx)
        tab = h[1].data
    ngroup = data.shape[0]

    def chip_wcs(i):
        w = WCS(naxis=2)
        w.wcs.crval = [tab["CRVAL1"][i], tab["CRVAL2"][i]]
        w.wcs.crpix = [tab["CRPIX1"][i], tab["CRPIX2"][i]]
        w.wcs.cd = [[tab["CD1_1"][i], tab["CD1_2"][i]],
                    [tab["CD2_1"][i], tab["CD2_2"][i]]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w

    off = np.linspace(-half_arcsec, half_arcsec, out_n)
    east, north = np.meshgrid(off, off)
    cosd = np.cos(np.deg2rad(dec_deg))
    ra = ra_deg - (east / 3600.0) / cosd
    dec = dec_deg + north / 3600.0
    ny, nx = data.shape[1], data.shape[2]
    mosaic = np.full((out_n, out_n), np.nan)
    for i in range(ngroup):
        px, py = chip_wcs(i).world_to_pixel_values(ra, dec)
        px = np.round(px).astype(int); py = np.round(py).astype(int)
        v = (px >= 0) & (px < nx) & (py >= 0) & (py < ny) & ~np.isfinite(mosaic)
        mosaic[v] = data[i][py[v], px[v]]
    meta = {"survey": "HST WFPC F785LP (waiver c0f)", "ok": True,
            "coverage_frac": float(np.isfinite(mosaic).mean())}
    return mosaic, half_arcsec, meta


def save_comparison(real_img, synth_img, half, save_path,
                    title="Real vs synthetic"):
    """Side-by-side real/synthetic plus a difference image (all normalised)."""
    import matplotlib.pyplot as plt

    def norm(a):
        a = np.array(a, dtype=float)
        finite = np.isfinite(a)
        if not finite.any() or a[finite].max() <= 0:
            return np.nan_to_num(a)
        return np.nan_to_num(a / a[finite].max())

    r, s = norm(real_img), norm(synth_img)
    diff = s - r
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, ttl, cmap in [(axes[0], r, "real (aligned)", "inferno"),
                               (axes[1], s, "synthetic", "inferno"),
                               (axes[2], diff, "synthetic - real", "coolwarm")]:
        im = ax.imshow(img, origin="lower", extent=[-half, half, -half, half],
                       cmap=cmap, vmin=(-1 if cmap == "coolwarm" else 0), vmax=1)
        ax.invert_xaxis(); ax.set_xlabel("E (arcsec)"); ax.set_ylabel("N (arcsec)")
        ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130); plt.close(fig)
    return save_path
