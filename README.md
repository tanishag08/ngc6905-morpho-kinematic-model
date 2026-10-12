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
| `main.py` | phase driver |

(Phases 4–8 add `observations.py`, `fitting.py`, `ages.py`, `viz3d.py`,
`export3d.py`, `editor.py`.)

## Assumptions vs published values

Published values (distance, inclination, PA, shell expansion velocity, systemic
velocity) are tagged `[PUBLISHED]` in `config.py` and are never changed
silently. Free parameters and image-estimated quantities (knot/lobe-tip
velocities, opening angle, thicknesses, knot size) are tagged `[ASSUMPTION]`.

## Status

- **Phase 1 — grid model & first renders:** complete.
- **Phase 2 — validation tests:** complete (7/7 pass).
- **Phase 3 — mesh + particle model:** complete (5/5 cross-checks; grid↔particle image corr 0.994).
- Phases 4–8: pending.
