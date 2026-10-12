"""
meshes.py -- SHAPE-style 3D mesh components (trimesh).

Primitives (sphere, ellipsoid, cone, cylinder, torus) plus the NGC 6905
component meshes, all built in the BODY frame (symmetry axis = body +y) and in
arcsec. Meshes are used for the interactive 3D view (Phase 7) and for export
(Phase 8); the quantitative physics rendering uses particles (particles.py).

Each component mesh is returned oriented so that:
  * the central shell is an ellipsoid centred on the star,
  * the two lobe cones have their apex at the star and open along +/- y,
  * the two knots are small spheres at the lobe tips (0, +/- p, 0).
"""

import numpy as np
import trimesh

import config as cfg


# --- Primitives ------------------------------------------------------------

def sphere(radius, subdivisions=3):
    """Unit-ish icosphere scaled to `radius` (arcsec), centred at origin."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def ellipsoid(semi_axes, subdivisions=3):
    """Ellipsoid with given (a_x, a_y, a_z) semi-axes, centred at origin."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    m.vertices = m.vertices * np.asarray(semi_axes, dtype=float)
    return m


def cone(half_angle_deg, height, sections=48, apex_at_origin=True, along_plus_y=True):
    """Cone of given half-opening angle and height, apex at origin, axis +/- y.

    trimesh builds a cone with base at z=0 and apex at z=height; we translate the
    apex to the origin and rotate the axis from +z onto +/- y.
    """
    base_radius = height * np.tan(np.deg2rad(half_angle_deg))
    m = trimesh.creation.cone(radius=base_radius, height=height, sections=sections)
    # trimesh cone: apex at z=height, base circle at z=0. Move apex to origin.
    if apex_at_origin:
        m.apply_translation([0, 0, -height])   # apex -> origin, base -> z=-height
    # Rotate so the base (now at -z) points to +y (or -y): map -z -> +/- y.
    sign = 1.0 if along_plus_y else -1.0
    # Rx(+90) maps -z -> +y ; Rx(-90) maps -z -> -y.
    ang = np.deg2rad(90.0) * sign
    R = trimesh.transformations.rotation_matrix(ang, [1, 0, 0])
    m.apply_transform(R)
    return m


def cylinder(radius, height, sections=48):
    """Cylinder centred at origin, axis along z (transform as needed)."""
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def torus(major_radius, minor_radius, sections=48):
    """Torus in the x-y plane centred at origin."""
    return trimesh.creation.torus(major_radius, minor_radius,
                                  major_sections=sections, minor_sections=sections // 2)


# --- NGC 6905 component meshes --------------------------------------------

def component_meshes(config=None):
    """Return {name: trimesh.Trimesh} for shell, both lobes, and both knots.

    All in the BODY frame (symmetry axis +y), arcsec.
    """
    if config is None:
        config = cfg.DEFAULT

    meshes = {}
    # Shell: ellipsoid at the mean shell radius (outer surface).
    meshes["shell"] = ellipsoid(config.shell.semi_axes_arcsec)

    # Lobes: two cones, apex at the star, opening along +/- y.
    meshes["lobe_N"] = cone(config.lobe.half_opening_angle_deg,
                            config.lobe.length_arcsec, along_plus_y=True)
    meshes["lobe_S"] = cone(config.lobe.half_opening_angle_deg,
                            config.lobe.length_arcsec, along_plus_y=False)

    # Knots: small spheres at the lobe tips.
    p = config.knot.axial_position_arcsec
    r = config.knot.sigma_arcsec * 1.5   # visual radius ~1.5 sigma
    kn = sphere(r); kn.apply_translation([0, p, 0]); meshes["knot_N"] = kn
    ks = sphere(r); ks.apply_translation([0, -p, 0]); meshes["knot_S"] = ks
    return meshes


def preview_meshes(meshes, save_path, title="NGC 6905 component meshes (body frame)"):
    """Save a static matplotlib 3D wireframe/surface preview of the meshes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import os

    colors = {"shell": "tab:cyan", "lobe_N": "tab:orange", "lobe_S": "tab:orange",
              "knot_N": "tab:red", "knot_S": "tab:red"}
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for name, m in meshes.items():
        tris = m.vertices[m.faces]
        col = Poly3DCollection(tris, alpha=0.25, facecolor=colors.get(name, "gray"),
                               edgecolor="none")
        ax.add_collection3d(col)
    lim = 45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("body x (arcsec)"); ax.set_ylabel("body y = axis (arcsec)")
    ax.set_zlabel("body z (arcsec)")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
