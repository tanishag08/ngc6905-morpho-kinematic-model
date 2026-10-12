"""
ages.py -- Phase 6: kinematic ages and the ejection history of NGC 6905.

KINEMATIC AGE
    t = (true 3-D distance from the star) / (expansion velocity)

What the inputs mean (stated explicitly, because the physics depends on it):
  * Angular sizes are treated as MEASURED ON THE SKY (projected).
  * Velocities are treated as true EXPANSION (space) velocities.

Deprojection with the inclination i (angle between the lobe axis and the plane
of the sky):
  * The central shell is ~spherical: its projected radius equals its true radius,
    so NO deprojection is applied (inclination is irrelevant for a sphere).
  * The bipolar lobes and the knots lie along the axis, so a feature at true
    distance R appears at projected separation p = R*cos(i). Hence the true
    distance is R = p / cos(i). At i = 60 deg this doubles the size.
Because the velocities are already the true expansion speeds, no velocity
deprojection is applied, and
    t_shell = R_shell / v_exp                      (i-independent)
    t_lobe  = (L_proj / cos i) / v_tip
    t_knot  = (p_proj / cos i) / v_knot

HOMOLOGOUS EXPANSION (see velocity.py) means each component has a single
well-defined kinematic age, so one number per component is meaningful.

LIMITATIONS (also printed and written to the summary):
  * Homologous (ballistic) expansion is an approximation; if gas has been
    accelerated by the stellar wind, the true age can be LONGER than the
    kinematic age (and vice-versa for decelerated gas).
  * The lobe-tip and knot velocities here are ASSUMED, not measured; the shell
    velocity is an older [O III] value. Real P-V spectroscopy is needed to pin
    them down.
  * Deprojection depends strongly on i (the 1/cos i factor), so the lobe/knot
    ages inherit a large inclination sensitivity.
"""

import os
import csv

import numpy as np

import config as cfg
from units import KM_PER_AU, SECONDS_PER_YEAR, AU_PER_PC, arcsec_to_au

RNG_SEED = cfg.DEFAULT.seed
N_MC = 10000

# Fractional 1-sigma velocity uncertainties: tighter for the published shell
# value, looser for the assumed lobe/knot velocities.
VEL_FRAC_PUBLISHED = 0.20
VEL_FRAC_ASSUMED = 0.40
SIZE_FRAC = 0.10            # +/-10% on all sizes
DIST_ERR_KPC = cfg.DEFAULT.geometry.distance_err_kpc   # +/-0.2 kpc
INCL_ERR_DEG = 10.0        # +/-10 deg
INCL_CLIP = (20.0, 85.0)   # keep 1/cos i and sin i well-behaved


def _component_specs(config):
    """Return the per-component inputs used for ages."""
    return [
        dict(name="shell",
             size_arcsec=config.shell.semi_axes_arcsec[0],   # roundish radius
             velocity_kms=config.shell.expansion_velocity_kms,
             bipolar=False, vel_frac=VEL_FRAC_PUBLISHED,
             size_note="PUBLISHED", vel_note="PUBLISHED [O III]"),
        dict(name="lobes",
             size_arcsec=config.lobe.length_arcsec,          # projected half-length
             velocity_kms=config.lobe.tip_velocity_kms,
             bipolar=True, vel_frac=VEL_FRAC_ASSUMED,
             size_note="PUBLISHED (extent)", vel_note="ASSUMPTION"),
        dict(name="knots",
             size_arcsec=config.knot.axial_position_arcsec,  # projected separation
             velocity_kms=config.knot.velocity_kms,
             bipolar=True, vel_frac=VEL_FRAC_ASSUMED,
             size_note="ASSUMPTION", vel_note="ASSUMPTION"),
    ]


def deproject_size_arcsec(size_arcsec, inclination_deg, bipolar):
    """True angular size: projected / cos(i) for bipolar features, else unchanged."""
    if not bipolar:
        return np.asarray(size_arcsec, dtype=float)
    return np.asarray(size_arcsec, dtype=float) / np.cos(np.deg2rad(inclination_deg))


def kinematic_age_yr(size_arcsec, velocity_kms, distance_pc, inclination_deg, bipolar):
    """Kinematic age (yr) from projected size, velocity, distance, inclination."""
    true_arcsec = deproject_size_arcsec(size_arcsec, inclination_deg, bipolar)
    r_au = arcsec_to_au(true_arcsec, distance_pc)
    return r_au * KM_PER_AU / (velocity_kms * SECONDS_PER_YEAR)


# ---------------------------------------------------------------------------
# Nominal table
# ---------------------------------------------------------------------------
def age_table(config=None):
    """Nominal per-component table. Returns a list of row dicts."""
    if config is None:
        config = cfg.DEFAULT
    d_pc = config.geometry.distance_kpc * 1000.0
    incl = config.geometry.inclination_deg
    rows = []
    for s in _component_specs(config):
        true_arcsec = float(deproject_size_arcsec(s["size_arcsec"], incl, s["bipolar"]))
        r_au = float(arcsec_to_au(true_arcsec, d_pc))
        r_pc = r_au / AU_PER_PC
        age = float(kinematic_age_yr(s["size_arcsec"], s["velocity_kms"], d_pc,
                                     incl, s["bipolar"]))
        rows.append(dict(
            component=s["name"],
            angular_proj_arcsec=round(s["size_arcsec"], 2),
            deprojected_arcsec=round(true_arcsec, 2),
            physical_AU=round(r_au, 1),
            physical_pc=round(r_pc, 4),
            velocity_kms=round(s["velocity_kms"], 1),
            age_yr=round(age, 0),
            bipolar=s["bipolar"],
            size_note=s["size_note"], vel_note=s["vel_note"],
        ))
    return rows


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(config=None, n=N_MC, seed=RNG_SEED):
    """Propagate uncertainties into the kinematic ages.

    Distance and inclination are sampled ONCE per iteration and SHARED by all
    components (they are common systematics for a single nebula). Each
    component's size and velocity are sampled independently. Returns a dict with
    per-component age-sample arrays plus the shared distance/inclination samples.
    """
    if config is None:
        config = cfg.DEFAULT
    rng = np.random.default_rng(seed)

    # Shared samples.
    d_pc = np.clip(rng.normal(config.geometry.distance_kpc, DIST_ERR_KPC, n), 0.1, None) * 1000.0
    incl = np.clip(rng.normal(config.geometry.inclination_deg, INCL_ERR_DEG, n),
                   *INCL_CLIP)

    out = {"distance_pc": d_pc, "inclination_deg": incl, "ages": {}}
    for s in _component_specs(config):
        size = np.clip(rng.normal(s["size_arcsec"], SIZE_FRAC * s["size_arcsec"], n), 1e-3, None)
        vel = np.clip(rng.normal(s["velocity_kms"], s["vel_frac"] * s["velocity_kms"], n), 1.0, None)
        ages = kinematic_age_yr(size, vel, d_pc, incl, s["bipolar"])
        out["ages"][s["name"]] = ages
    return out


def summarize_mc(mc):
    """Median and 1-sigma (16th-84th percentile) for each component (yr)."""
    stats = {}
    for name, ages in mc["ages"].items():
        lo, med, hi = np.percentile(ages, [16, 50, 84])
        stats[name] = dict(median=float(med), lo=float(lo), hi=float(hi),
                           minus=float(med - lo), plus=float(hi - med))
    return stats


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_age_distributions(mc, save_path, stats=None):
    import matplotlib.pyplot as plt
    if stats is None:
        stats = summarize_mc(mc)
    colors = {"shell": "tab:cyan", "lobes": "tab:orange", "knots": "tab:red"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    allages = np.concatenate([a for a in mc["ages"].values()])
    bins = np.linspace(0, np.percentile(allages, 99), 60)
    for name, ages in mc["ages"].items():
        ax.hist(ages, bins=bins, histtype="stepfilled", alpha=0.45,
                color=colors.get(name), density=True,
                label=f"{name}: {stats[name]['median']:,.0f} "
                      f"(+{stats[name]['plus']:,.0f}/-{stats[name]['minus']:,.0f}) yr")
        ax.axvline(stats[name]["median"], color=colors.get(name), ls="--", lw=1.2)
    ax.set_xlabel("kinematic age (yr)")
    ax.set_ylabel("probability density")
    ax.set_title("NGC 6905 kinematic ages (10,000-sample Monte Carlo)\n"
                 "shell spherical; lobes/knots deprojected by 1/cos i  "
                 "(assumed velocities flagged)")
    ax.legend(title="median (1σ)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
def interpret(mc, stats=None):
    """Build a text interpretation of the ejection history."""
    if stats is None:
        stats = summarize_mc(mc)
    ages = mc["ages"]
    order = sorted(stats, key=lambda k: stats[k]["median"], reverse=True)  # oldest first

    lines = []
    lines.append("Ejection order (oldest -> youngest, by median kinematic age):")
    for k in order:
        s = stats[k]
        lines.append(f"  {k:6s}: {s['median']:,.0f} yr "
                     f"(1σ {s['lo']:,.0f}–{s['hi']:,.0f})")

    # Pairwise significance using the CORRELATED samples (shared distance/incl.).
    lines.append("\nAre the ages significantly different? "
                 "(fraction of MC samples with A older than B)")
    pairs = [("lobes", "shell"), ("knots", "shell"), ("lobes", "knots")]
    for a, b in pairs:
        frac = float(np.mean(ages[a] > ages[b]))
        dmed = np.median(ages[a] - ages[b])
        dlo, dhi = np.percentile(ages[a] - ages[b], [16, 84])
        verdict = ("clearly older" if frac > 0.95 else
                   "clearly younger" if frac < 0.05 else
                   "NOT clearly distinguishable")
        lines.append(f"  {a} vs {b}: P({a}>{b})={frac:.2f} -> {a} is {verdict}; "
                     f"Δt median={dmed:,.0f} yr (1σ {dlo:,.0f}..{dhi:,.0f})")

    # Narrative.
    shell_young = stats["shell"]["median"] < min(stats["lobes"]["median"],
                                                 stats["knots"]["median"])
    lines.append("")
    if shell_young:
        lines.append("Interpretation: the central shell is the YOUNGEST component, while the")
        lines.append("deprojected bipolar lobes/knots have longer kinematic timescales. Taken at")
        lines.append("face value this points to an EARLIER large-scale bipolar ejection followed")
        lines.append("by a more recent central shell -- i.e. SEPARATE ejection episodes, the")
        lines.append("scenario motivating this study. HOWEVER (see limitations) the lobe/knot ages")
        lines.append("are inflated by the 1/cos i deprojection and rest on ASSUMED velocities, so")
        lines.append("the significance must be read together with the Monte-Carlo overlaps above.")
    else:
        lines.append("Interpretation: the components' ages overlap within their (large)")
        lines.append("uncertainties, so a single ejection episode cannot be ruled out.")
    lines.append("")
    lines.append("LIMITATIONS: homologous expansion is an approximation (wind acceleration would")
    lines.append("make true ages older); lobe/knot velocities are ASSUMED and the shell velocity")
    lines.append("is an old [O III] value; ages depend strongly on inclination via 1/cos i.")
    lines.append("Fitted P-V spectroscopy is required to turn these into firm ages.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write_csv(rows, stats, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fields = ["component", "angular_proj_arcsec", "deprojected_arcsec",
              "physical_AU", "physical_pc", "velocity_kms", "age_yr",
              "mc_median_yr", "mc_lo_yr", "mc_hi_yr", "size_note", "vel_note"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            name = r["component"]
            row = dict(r)
            row["mc_median_yr"] = round(stats[name]["median"], 0)
            row["mc_lo_yr"] = round(stats[name]["lo"], 0)
            row["mc_hi_yr"] = round(stats[name]["hi"], 0)
            row.pop("bipolar", None)
            w.writerow({k: row.get(k) for k in fields})
    return save_path


def run(config=None, outputs_dir=None):
    if config is None:
        config = cfg.DEFAULT
    out = outputs_dir or config.outputs_dir
    rows = age_table(config)
    mc = monte_carlo(config)
    stats = summarize_mc(mc)

    # Print table.
    print(f"{'component':8s} {'proj″':>7s} {'deproj″':>8s} {'AU':>9s} {'pc':>7s} "
          f"{'v(km/s)':>8s} {'age(yr)':>9s}  {'MC median (1σ)':>22s}")
    for r in rows:
        s = stats[r["component"]]
        print(f"{r['component']:8s} {r['angular_proj_arcsec']:7.1f} "
              f"{r['deprojected_arcsec']:8.1f} {r['physical_AU']:9.0f} "
              f"{r['physical_pc']:7.3f} {r['velocity_kms']:8.1f} {r['age_yr']:9.0f}  "
              f"{s['median']:8.0f} (+{s['plus']:.0f}/-{s['minus']:.0f})")

    csv_path = write_csv(rows, stats, f"{out}/phase6_ages_table.csv")
    plot_path = plot_age_distributions(mc, f"{out}/phase6_age_distributions.png", stats)
    text = interpret(mc, stats)
    print("\n" + text)
    with open(f"{out}/phase6_age_interpretation.txt", "w") as f:
        f.write(text + "\n")
    print(f"\n[saved] {csv_path}\n[saved] {plot_path}\n[saved] {out}/phase6_age_interpretation.txt")
    return rows, mc, stats


if __name__ == "__main__":
    run()
