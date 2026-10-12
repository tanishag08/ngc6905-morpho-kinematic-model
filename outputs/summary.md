# NGC 6905 — SHAPE-style 3D morpho-kinematic model: summary

A reproducible Python toolkit that reconstructs the 3-D structure and velocity
field of the planetary nebula **NGC 6905** (the "Blue Flash Nebula") in the
style of the *SHAPE* software, estimates the kinematic ages of its components,
and compares the model to a real image.

---

## 1. The object

NGC 6905 is a planetary nebula in Delphinus powered by a hot [WO]-type
Wolf–Rayet central star (HD 193949, T_eff ≈ 140 kK, wind ≈ 2000 km/s). It has
three morphological components:

1. a roundish/ellipsoidal **central shell** (angular radius ≈ 19″; literature
   inner shell ≈ 47×34″),
2. a pair of V-shaped **bipolar lobes** (total extent ≈ 80″, PA ≈ 160°), and
3. a pair of low-ionization **knots (ansae)** at the lobe tips, bright in [N II].

No 3-D morpho-kinematic model of NGC 6905 had been published; this project
builds one and uses it to ask whether the shell, lobes, and knots represent
**separate ejection episodes**.

---

## 2. The model

**Representation.** Two interchangeable, cross-validated representations:
a 128³ analytic emissivity **grid** (Phase 1) and a SHAPE-style **particle**
cloud (200,000 particles/component) filling trimesh primitives with stackable
modifiers (Phase 3). The two agree to image correlation 0.994 / P-V 0.986.

**Components (body frame, symmetry axis = body +y):**
| component | shape | key parameters |
|-----------|-------|----------------|
| shell | ellipsoidal shell | radius 19″, thickness 4″ |
| lobes | two hollow cones | half-length 40″, half-opening 18°, wall 4″ |
| knots | two Gaussian blobs | at ±40″, σ = 4″ |

**Geometry / orientation.** Sky frame x=East (left), y=North (up), z=line of
sight (positive away → redshift). Inclination i = 60° (axis vs. plane of sky),
PA = 160° (N through E). Body→sky rotation R(i,PA)=Rz(−PA)·Rx(i).

**Velocity field.** Homologous (Hubble-type) expansion v = H·r, with each
component's H set so it reaches its characteristic velocity at its
characteristic radius. Homologous expansion ⇒ one kinematic age per component.
Line-of-sight velocity simplifies to v_los = H·z.

**Validation (Phase 2).** 7/7 tests pass: expanding-shell P-V ellipse; inclined-
ring P-V line (slope exact); lobe symmetry at i=0; lobe overlap+velocity split at
i=90; flux conservation under rotation (0.27%); unit checks (2700 AU/″, 211 AU);
shell age sanity (5404 yr).

---

## 3. Parameters (published vs. assumed)

Published values are never changed silently; assumptions are flagged everywhere.

| parameter | value | source |
|-----------|-------|--------|
| distance | 2.7 ± 0.2 kpc | **PUBLISHED** (Gaia, Bailer-Jones+ 2021) |
| inclination | 60° | **PUBLISHED** (Cuesta+ 1993) |
| position angle | 160° | **PUBLISHED** |
| shell radius | 19″ | **PUBLISHED** |
| shell v_exp | 45 km/s | **PUBLISHED** [O III] (½ of the ~100 km/s splitting) |
| systemic velocity | −8 km/s | **PUBLISHED** (subtracted) |
| lobe half-opening | 18° | *assumption* (from image) |
| shell/lobe thickness, knot size | 4″ | *assumption* (from image) |
| lobe tip velocity | 60 km/s | *assumption* (free parameter) |
| knot velocity | 70 km/s | *assumption* (free parameter) |

---

## 4. Comparison with a real image (Phase 4)

A **Pan-STARRS1** optical stack (from MAST) was aligned to the model grid
(north-up/east-left). The **g (contains [O III]+Hβ) and r (Hα+[N II]) bands
clearly resolve the nebula**; an HST WFPC-1 F785LP image also found at MAST does
**not** show the nebula (red long-pass blocks all the bright lines), and the
STIS/FUSE/TESS products are central-star only. (The FUSE spectrum does show the
O VI λ1032,1038 wind P Cygni feature to ≈ −2000 km/s — the fast [WO] wind
invoked to shape the lobes.)

**Result of the image fit:** the published **shell radius (19→19.3″)** and
**lobe length (~38″)** are confirmed by the real nebula, and the bright shell
axis ratio b/a ≈ 0.78 matches the literature 47×34″. The PA/inclination are
**not** reliably constrained by broadband imaging with a thin-shell model
(correlation 0.64): the broadband image shows the *filled* ionized nebula, not
the thin emitting shells. **Image fitting constrains geometry only — never the
velocities or ages.**

---

## 5. Kinematic ages and ejection history (Phase 6)

Age = (true 3-D distance)/(expansion velocity). The shell is ~spherical (no
deprojection); the bipolar lobes/knots are deprojected by 1/cos i. 10,000-sample
Monte Carlo with distance (±0.2 kpc) and inclination (±10°) shared across
components, and per-component size (±10%) and velocity (±20% published / ±40%
assumed).

| component | deprojected size | physical | v (km/s) | **kinematic age (median, 1σ)** |
|-----------|------------------|----------|----------|-------------------------------|
| shell | 19″ | 0.25 pc | 45 | **5,400 (+1,600/−1,100) yr** |
| knots | 80″ | 1.05 pc | 70 | **15,900 (+15,400/−6,300) yr** |
| lobes | 80″ | 1.05 pc | 60 | **18,400 (+17,400/−7,100) yr** |

**Ejection history.** The central **shell is clearly the youngest** component:
P(lobes older than shell) = 0.99, P(knots older than shell) = 0.98, a median gap
of ~10–13 kyr. The **lobes and knots are indistinguishable** (P = 0.59). Taken at
face value this favours an **earlier large-scale bipolar ejection (~16–18 kyr)
followed by a more recent central shell (~5 kyr)** — i.e. *separate ejection
episodes*, the scenario motivating the study.

---

## 6. Limitations

- **Homologous expansion is an approximation.** If gas has been accelerated by
  the stellar wind, true ages are *longer* than kinematic ages (and vice-versa
  for deceleration).
- **Velocities are old/assumed.** The shell v_exp is an older [O III] value; the
  lobe-tip and knot velocities are *assumptions* (±40%). This dominates the large
  lobe/knot age uncertainties.
- **Strong inclination sensitivity.** Lobe/knot ages scale as 1/cos i, so the
  ±10° inclination uncertainty produces the long right-skewed tails.
- **Imaging cannot constrain velocities.** The real Pan-STARRS image anchors the
  *geometry/size*; the 3-D orientation and velocities — and therefore firm ages
  — require **long-slit P-V spectroscopy**, which this project is built to ingest
  (the renderers and P-V machinery are ready for real FITS P-V data).

---

## 7. Outputs (in `outputs/`)

Conventions diagram · Phase-2 validation plots · grid↔particle cross-check ·
synthetic image/composite/channel maps · smoothed P-V (both slits) · real
Pan-STARRS vs. synthetic + fit · FUSE spectrum · kinematic-age table (CSV) +
distribution plot + interpretation · interactive 3-D HTML · rotation &
expansion GIFs · 3-D model files (`ngc6905.glb`, `.obj`, watertight `.stl` ×4
scaled to 10 cm). Best-fit parameters in `phase4_ps1fit_params.json`.

Run any phase with `python main.py phaseN` (or `python main.py all`); launch the
interactive editor with `python editor.py`.

---

## 8. References

- Sabbadin & Hamzaoglu 1982, A&AS, 50, 1
- Cuesta, Phillips & Mampaso 1993, A&A, 267, 199
- Phillips & Ramos-Larios 2010, MNRAS, 405, 2179
- Gómez-González et al. 2022, MNRAS, 509, 974 (arXiv:2110.09551)
- Mari, Gonçalves & Akras 2023, MNRAS, 518, 3908
- Steffen et al. 2011 (SHAPE method), IEEE TVCG, 17, 454
- Bailer-Jones et al. 2021, AJ, 161, 147 (Gaia EDR3 distances)
