"""
export3d.py -- Phase 8: export the 3D model to standard mesh files.

Exports the BODY-FRAME model (the intrinsic 3-D structure; orientation i/PA is
only a viewing direction, so the physical object lives in the body frame):

  * .glb : glTF binary -- browsers, phones, iPhone AR Quick Look, Blender.
           A scene with each component as a SEPARATE, COLOURED object.
  * .obj : general 3-D software -- scene with separate coloured objects (+ .mtl).
  * .stl : 3-D printing -- a single WATERTIGHT solid scaled to fit a 10 cm cube,
           plus one STL per component (STL carries no colour).

Each exported file is verified by loading it back with trimesh.

Component grouping / colours:
  shell (cyan), lobes = two cones (orange), knots = two spheres (red).
"""

import os

import numpy as np
import trimesh

import config as cfg
from config import ID_SHELL, ID_LOBE, ID_KNOT
from meshes import component_meshes

# RGBA face colours (0-255) per component.
COMPONENT_RGBA = {
    "shell": [51, 181, 255, 255],
    "lobes": [255, 154, 26, 255],
    "knots": [255, 51, 51, 255],
}

PRINT_SIZE_MM = 100.0     # STL: largest dimension -> 10 cm
VIEW_SIZE_M = 0.15        # GLB/OBJ: largest dimension -> 15 cm (metres)


# ---------------------------------------------------------------------------
# Build grouped, coloured component meshes (body frame)
# ---------------------------------------------------------------------------
def build_component_meshes(config=None):
    """Return {component_name: trimesh.Trimesh} with colours set.

    The two lobe cones are merged into 'lobes' and the two knot spheres into
    'knots'. Each group is watertight (a concatenation of closed meshes).
    """
    if config is None:
        config = cfg.DEFAULT
    raw = component_meshes(config)
    groups = {
        "shell": [raw["shell"]],
        "lobes": [raw["lobe_N"], raw["lobe_S"]],
        "knots": [raw["knot_N"], raw["knot_S"]],
    }
    out = {}
    for name, parts in groups.items():
        m = trimesh.util.concatenate([p.copy() for p in parts])
        m.visual.face_colors = COMPONENT_RGBA[name]
        out[name] = m
    return out


def _combined_extent(meshes):
    """Largest bounding-box dimension across all meshes (same units in/out)."""
    mins = np.min([m.bounds[0] for m in meshes.values()], axis=0)
    maxs = np.max([m.bounds[1] for m in meshes.values()], axis=0)
    return float(np.max(maxs - mins))


def scale_meshes(meshes, target_size):
    """Return copies scaled so the combined largest dimension == target_size."""
    extent = _combined_extent(meshes)
    factor = target_size / extent if extent > 0 else 1.0
    out = {}
    for name, m in meshes.items():
        c = m.copy()
        c.apply_scale(factor)
        out[name] = c
    return out, factor


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------
def export_scene(meshes, path):
    """Export a scene (separate named coloured objects) to .glb or .obj."""
    scene = trimesh.Scene()
    for name, m in meshes.items():
        scene.add_geometry(m, geom_name=name, node_name=name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.export(path)
    return path


def make_watertight_solid(meshes):
    """Combine components into one solid for printing.

    Tries a boolean union (needs a backend); falls back to concatenation of the
    individually-watertight components (interpenetrating closed shells, which
    slicers treat as a union).
    """
    parts = [m.copy() for m in meshes.values()]
    try:
        solid = trimesh.boolean.union(parts)
        if solid is not None and solid.is_watertight and len(solid.faces) > 0:
            return solid, "boolean union"
    except Exception:
        pass
    return trimesh.util.concatenate(parts), "concatenation (per-part watertight)"


def export_stls(meshes, out_dir, basename="ngc6905"):
    """Export a combined watertight STL (10 cm) + one STL per component."""
    scaled, factor = scale_meshes(meshes, PRINT_SIZE_MM)
    paths = {}
    # per-component
    for name, m in scaled.items():
        p = f"{out_dir}/{basename}_{name}.stl"
        m.export(p)
        paths[name] = p
    # combined
    solid, method = make_watertight_solid(scaled)
    combined_path = f"{out_dir}/{basename}_combined.stl"
    solid.export(combined_path)
    paths["combined"] = combined_path
    return paths, factor, method


# ---------------------------------------------------------------------------
# Verification (load each file back with trimesh)
# ---------------------------------------------------------------------------
def verify(path):
    """Load a file back and return a short status string."""
    try:
        obj = trimesh.load(path, force="scene" if path.endswith((".glb", ".obj")) else None)
        if isinstance(obj, trimesh.Scene):
            n_geo = len(obj.geometry)
            n_faces = sum(len(g.faces) for g in obj.geometry.values())
            return f"OK scene: {n_geo} objects, {n_faces} faces"
        else:
            return (f"OK mesh: {len(obj.faces)} faces, "
                    f"watertight={obj.is_watertight}, volume={obj.is_volume}")
    except Exception as exc:
        return f"FAILED to load back: {exc!r}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(config=None, outputs_dir=None):
    if config is None:
        config = cfg.DEFAULT
    out = outputs_dir or config.outputs_dir
    os.makedirs(out, exist_ok=True)

    meshes = build_component_meshes(config)
    for name, m in meshes.items():
        print(f"  {name:6s}: {len(m.vertices):5d} verts, {len(m.faces):5d} faces, "
              f"watertight={m.is_watertight}")

    # GLB and OBJ scenes scaled for viewing/AR (metres).
    view_meshes, vfac = scale_meshes(meshes, VIEW_SIZE_M)
    glb = export_scene(view_meshes, f"{out}/ngc6905.glb")
    obj = export_scene(view_meshes, f"{out}/ngc6905.obj")
    print(f"\n[saved] {glb}   (scaled x{vfac:.2e} -> {VIEW_SIZE_M*100:.0f} cm)")
    print(f"[saved] {obj}")

    # STLs scaled to a 10 cm print (mm).
    stl_paths, sfac, method = export_stls(meshes, out)
    print(f"[saved] STLs (scaled x{sfac:.2e} -> {PRINT_SIZE_MM/10:.0f} cm; "
          f"combined via {method}):")
    for name, p in stl_paths.items():
        print(f"         {p}")

    # Verify every file loads back.
    print("\nVerification (load back with trimesh):")
    for p in [glb, obj] + list(stl_paths.values()):
        print(f"  {os.path.basename(p):28s} -> {verify(p)}")
    return {"glb": glb, "obj": obj, "stl": stl_paths}


if __name__ == "__main__":
    run()
