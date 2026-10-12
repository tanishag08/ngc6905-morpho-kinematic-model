"""
modifiers.py -- SHAPE-style stackable mesh/particle modifiers.

In SHAPE you start from a primitive and apply a stack of modifiers that deform
it. Here a *geometric* modifier is simply a callable that maps an (N,3) array of
points to a new (N,3) array. Because mesh vertices and particle positions are
both just (N,3) point sets, the SAME modifier object can deform a mesh and the
particles sampled from it -- which keeps the two representations consistent.

Geometric modifiers (position -> position):
  * Scale    -- squash/stretch along an axis (factor < 1 squashes, > 1 stretches)
  * Twist    -- rotate about an axis by an angle proportional to the axial coord
  * Bump     -- sinusoidal (azimuthal) surface corrugation about an axis
  * Taper    -- perpendicular radius changes linearly along an axis

Emissivity modifier (position -> weight multiplier):
  * DensityFalloff -- emissivity decreasing from a reference surface inward,
                      i.e. shell-thickness control for a filled volume.

Modifiers are applied in the order they appear in a ModifierStack.

Axes are given as integer indices 0, 1, 2 == body x, y, z. The nebular symmetry
axis is body +y, so most lobe modifiers use axis=1.
"""

import numpy as np
import trimesh

# Perpendicular axis pairs for each axis index (used by Twist/Bump/Taper).
_PERP = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


class Scale:
    """Squash/stretch: multiply one axis by a constant factor."""

    def __init__(self, axis, factor):
        self.axis = axis
        self.factor = factor

    def __call__(self, pts):
        pts = np.array(pts, dtype=float, copy=True)
        pts[:, self.axis] *= self.factor
        return pts


class Twist:
    """Twist about an axis: rotation angle grows with the axial coordinate.

    rate_deg_per_arcsec gives degrees of rotation per arcsec along `axis`.
    Points are rotated about `axis` by angle = rate * (axial coordinate), so the
    structure winds up like a corkscrew.
    """

    def __init__(self, axis, rate_deg_per_arcsec):
        self.axis = axis
        self.rate = np.deg2rad(rate_deg_per_arcsec)

    def __call__(self, pts):
        pts = np.array(pts, dtype=float, copy=True)
        j, k = _PERP[self.axis]
        s = pts[:, self.axis]
        ang = self.rate * s
        c, sn = np.cos(ang), np.sin(ang)
        pj, pk = pts[:, j].copy(), pts[:, k].copy()
        pts[:, j] = c * pj - sn * pk
        pts[:, k] = sn * pj + c * pk
        return pts


class Bump:
    """Azimuthal sinusoidal corrugation of the perpendicular radius about an axis.

    r_perp -> r_perp + amplitude * sin(frequency * phi), where phi is the azimuth
    around `axis`. `frequency` is the number of bumps around the circumference.
    Amplitude is in arcsec.
    """

    def __init__(self, axis, amplitude_arcsec, frequency):
        self.axis = axis
        self.amplitude = amplitude_arcsec
        self.frequency = frequency

    def __call__(self, pts):
        pts = np.array(pts, dtype=float, copy=True)
        j, k = _PERP[self.axis]
        pj, pk = pts[:, j], pts[:, k]
        r = np.hypot(pj, pk)
        phi = np.arctan2(pk, pj)
        delta = self.amplitude * np.sin(self.frequency * phi)
        scale = np.where(r > 1e-9, (r + delta) / r, 1.0)
        pts[:, j] = pj * scale
        pts[:, k] = pk * scale
        return pts


class Taper:
    """Linear change of perpendicular radius along an axis.

    The perpendicular radius is scaled by a factor that varies linearly from
    `start_scale` at the low end of the axial range to `end_scale` at the high
    end. If use_abs is True the taper is symmetric in |axial coord| (useful for a
    two-sided bipolar structure).
    """

    def __init__(self, axis, start_scale, end_scale, use_abs=False):
        self.axis = axis
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.use_abs = use_abs

    def __call__(self, pts):
        pts = np.array(pts, dtype=float, copy=True)
        j, k = _PERP[self.axis]
        s = np.abs(pts[:, self.axis]) if self.use_abs else pts[:, self.axis]
        s_min, s_max = s.min(), s.max()
        t = np.zeros_like(s) if s_max == s_min else (s - s_min) / (s_max - s_min)
        scale = self.start_scale + (self.end_scale - self.start_scale) * t
        pts[:, j] *= scale
        pts[:, k] *= scale
        return pts


class SizeCurve:
    """SHAPE-style 'Size' / 'Squeeze' modifier driven by a curve.

    In SHAPE the Size modifier scales an object's radius along its axis according
    to a curve edited in the Graphs module. Here `control_scales` are the radius
    scale factors at normalised axial positions linspace(0, 1, N) (0 = base /
    object centre, 1 = tip); intermediate values are linearly interpolated. The
    perpendicular radius of each point is multiplied by the interpolated scale.
    """

    def __init__(self, axis, control_scales, use_abs=True):
        self.axis = axis
        self.scales = np.asarray(control_scales, dtype=float)
        self.use_abs = use_abs

    def __call__(self, pts):
        pts = np.array(pts, dtype=float, copy=True)
        j, k = _PERP[self.axis]
        s = np.abs(pts[:, self.axis]) if self.use_abs else pts[:, self.axis]
        s_min, s_max = s.min(), s.max()
        t = np.zeros_like(s) if s_max == s_min else (s - s_min) / (s_max - s_min)
        xp = np.linspace(0.0, 1.0, len(self.scales))
        scale = np.interp(t, xp, self.scales)
        pts[:, j] *= scale
        pts[:, k] *= scale
        return pts


class DensityFalloff:
    """Emissivity weighting that decreases from a reference surface inward.

    Given each point's normalized radius q (q = 1 on the surface, q < 1 inside),
    return a weight multiplier. The profile is a Gaussian in (1 - q):
        w = exp( -((1 - q) / width)^2 )
    so emission concentrates in a shell of fractional thickness ~ `width` just
    inside the surface. This is how a filled mesh is turned into a shell.
    """

    def __init__(self, width):
        self.width = width

    def weights(self, q_normalized):
        q = np.asarray(q_normalized, dtype=float)
        return np.exp(-((1.0 - q) / self.width) ** 2)


class ModifierStack:
    """An ordered list of geometric modifiers applied in sequence."""

    def __init__(self, modifiers=None):
        self.modifiers = list(modifiers) if modifiers else []

    def __call__(self, pts):
        for m in self.modifiers:
            pts = m(pts)
        return pts

    def apply_to_mesh(self, mesh):
        """Return a copy of a trimesh mesh with the stack applied to its vertices."""
        m = mesh.copy()
        m.vertices = self(np.asarray(m.vertices, dtype=float))
        return m

    def is_empty(self):
        return len(self.modifiers) == 0
