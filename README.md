# NGC 6905 — SHAPE-style 3D morpho-kinematic model

A Python toolkit that reconstructs the 3D structure and velocity field of the
planetary nebula **NGC 6905** (the "Blue Flash Nebula") the way the *SHAPE*
software (Steffen & López) does: analytic/mesh components, particle rendering,
synthetic images, position–velocity (P-V) diagrams, channel maps, an
interactive editor, 3D exports, and kinematic-age estimates.

This is a learning project built **phase by phase**. Each phase is verified
before the next begins.

## Install

```bash
cd ngc6905_model
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Requires Python 3.11+ (developed on 3.13).

## Run

```bash
python main.py phase1      # build the analytic grid model + first observables
python main.py phase2      # run the 7 validation tests (writes outputs/phase2_*.png)
python main.py phase3      # mesh + particle model + modifier demo + cross-check
python main.py phase4      # synthetic obs (seeing, channels, composite) + real image + fit
python main.py phase5      # self-test the interactive editor
python editor.py           # launch the browser editor at http://127.0.0.1:8050
python main.py phase6      # kinematic ages + Monte Carlo + ejection history
python main.py phase7      # interactive 3D HTML + rotation GIF + expansion GIF
python main.py phase8      # export .glb / .obj / watertight .stl (+ verify)
python main.py all         # run the whole non-interactive pipeline
```

### Supplying a real image yourself

Phase 4 tries to download a DSS/Pan-STARRS cutout via SkyView. If that fails
(no network, or SkyView is slow — each survey is time-bounded so it never
hangs), supply a local FITS instead:

```python
import observations as obs
real, half, meta = obs.load_fits_image("my_ngc6905.fits")   # aligned to the model grid
```

All figures and tables are written to `outputs/`.

## Conventions (see `orientation.py`)

- Origin at the central star.
- Sky frame: **x = east** (left on images), **y = north** (up), **z = line of
  sight, positive away from the observer** → positive velocity = **redshift**.
- Inclination `i` = angle between the lobe axis and the plane of the sky.
- Position angle `PA` measured from **north through east**.
- Body→sky rotation: `R(i, PA) = Rz(−PA) · Rx(i)`, with each component built
  with its symmetry axis along **body +y**.

See `outputs/phase1_conventions.png` for the labelled diagram.

## Module map

| file | role |
|------|------|
| `config.py` | all parameters, units, and labelled assumptions |
| `units.py` | arcsec/AU/pc/km/km·s⁻¹/yr conversions and kinematic age |
| `orientation.py` | rotations, projection, conventions diagram |
| `velocity.py` | homologous and power-law velocity fields |
| `grid_model.py` | Phase 1 analytic emissivity grid |
| `render.py` | sky images and P-V diagrams (grid + particle) |
| `meshes.py` | trimesh primitives + NGC 6905 component meshes |
| `modifiers.py` | stackable SHAPE modifiers (squash, twist, bump, taper, density falloff) |
| `particles.py` | particle sampling, velocities, model builder |
| `tests.py` | Phase 2 validation + Phase 3 particle cross-check |
| `observations.py` | download/align a real image (SkyView/SIMBAD) or load a FITS |
| `fitting.py` | image-based geometry fitting (scipy.optimize) |
| `editor.py` | interactive browser editor (Plotly Dash) |
| `ages.py` | kinematic ages, Monte Carlo, ejection history |
| `viz3d.py` | interactive 3D HTML + rotation/expansion GIFs |
| `export3d.py` | export .glb / .obj / .stl + verify |
| `main.py` | phase driver |

## Assumptions vs published values

Published values (distance, inclination, PA, shell expansion velocity, systemic
velocity) are tagged `[PUBLISHED]` in `config.py` and are never changed
silently. Free parameters and image-estimated quantities (knot/lobe-tip
velocities, opening angle, thicknesses, knot size) are tagged `[ASSUMPTION]`.

## Status

- **Phase 1 — grid model & first renders:** complete.
- **Phase 2 — validation tests:** complete (7/7 pass).
- **Phase 3 — mesh + particle model:** complete (5/5 cross-checks; grid↔particle image corr 0.994).
- **Phase 4 — synthetic obs + real image + fitting:** complete. Synthetic image
  (seeing), colour composite, instrumentally-smoothed P-V, channel maps, and an
  image fitter (self-consistency corr 0.999) all work. The real-image download
  depends on SkyView being reachable; it is time-bounded and degrades to a FITS
  fallback. Image fitting constrains geometry only — **not** velocities/ages.
  **Real data:** a Pan-STARRS g/r image of NGC 6905 (from MAST) is resolved,
  aligned, and compared/fit (`outputs/phase4_real_vs_synthetic_PS1.png`).
- **Phase 5 — interactive editor:** complete. `editor.py` is a Plotly Dash app
  with sliders for every parameter, live 3D + synthetic/real image + both P-V
  diagrams + channel maps + live kinematic ages, and save/load/reset.
- **Phase 6 — kinematic ages & history:** complete. Deprojected ages
  (shell ~5.4 kyr; lobes/knots ~16–18 kyr), 10,000-sample Monte Carlo, age
  distribution plot, CSV table, and interpretation. Shell is clearly the
  youngest (P≈0.98–0.99); lobes vs knots indistinguishable. Big caveats:
  assumed velocities + 1/cos i deprojection → P-V spectroscopy needed.
- **Phase 7 — 3D visualization:** complete. Standalone interactive Plotly HTML
  (toggles: particles by component / by v_los, mesh surfaces, velocity arrows;
  LOS arrow + AU scale bar), a 360° rotation GIF, and a homologous expansion
  time-lapse GIF (each component starts at its own ejection time).
- **Phase 8 — 3D export:** complete. `ngc6905.glb` (3 separate coloured objects
  for AR/Blender), `.obj`, and watertight `.stl` files (combined + per-component)
  scaled to a 10 cm print; every file verified by loading back with trimesh.
- **Final `outputs/summary.md`** ties together the model, real-image comparison,
  ages, interpretation, limitations, and references.

**All 8 phases complete.**
