"""One signed number per contrast: the fraction of identity information lost.

step4 scores UL and UH separately, because the excess score needs a sign convention per
direction. The information-loss score does not - dlogit already carries the sign - so UL
and UH are two halves of one quantity and belong in one number, each weighted by its
share of the panel's identity information:

    fraction lost = sum_d frac_d * J_d / sum_d J_d

0 = unchanged, 1 = collapsed onto a generic cell, < 0 = diverged away from it. The
combined per-sample values are then tested the same way the design calls for. The
headline frac_lost is net of the matched background (step4); frac_lost_raw and
frac_lost_bg are its two components, combined the same way, and frac_lost_raw is tested
too so the effect of the subtraction is visible. mde80 is the smallest loss the design
detects with 80% power at two-sided alpha 0.05 (normal approximation; observed per-pair
SD, or pooled SD for unpaired designs).

frac_lost_low/_high and p_low/_high are the UL and UH halves before combining (net of
background, from contrast_tests.csv), so a direction with little Jeffreys mass - the 10
identity UH CpGs carry under 2% of that layer's - stays visible.

Also writes the chronic-radiation arm adjusted for population doublings: those cultures
were passaged while irradiated, so the untreated passaging series of the same experiment
is interpolated to each irradiated culture's doubling level and subtracted. raw_shift is
the same contrast without that adjustment, against the untreated culture of the same
passage.

Writes results/scores/information_loss_summary.csv, per_sample_information_loss.csv (the
combined per-sample values behind it) and chronic_radiation_pdl_adjusted.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common

Z80 = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.8))

KEY = ["dataset", "contrast", "panel_base", "layer_split", "layer", "age"]


def split_panel(df):
    df = df.copy()
    df["panel_base"] = df.panel.str.split(":").str[0]
    df["layer_split"] = df.panel.str.split(":").str[1]
    return df


def bootstrap_ci(v, ctrl=None, n_boot=10000, seed=3):
    """Percentile bootstrap 95% CI on the mean loss. Unpaired designs resample both groups."""
    if len(v) < 3 or (ctrl is not None and len(ctrl) < 3):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    if ctrl is not None:
        draws = draws - rng.choice(ctrl, size=(n_boot, len(ctrl)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def mde80(v, ctrl=None):
    """Minimum detectable effect: paired if ctrl is None, else two groups with pooled SD."""
    if ctrl is None:
        return Z80 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    if min(len(v), len(ctrl)) < 2:
        return np.nan
    sp = np.sqrt(((len(v) - 1) * np.var(v, ddof=1) + (len(ctrl) - 1) * np.var(ctrl, ddof=1))
                 / (len(v) + len(ctrl) - 2))
    return Z80 * sp * np.sqrt(1 / len(v) + 1 / len(ctrl))


def chronic_radiation_vs_passaging():
    """Chronic-radiation shift with the untreated passaging series subtracted at matched doublings."""
    ser = pd.read_csv(common.SCORES / "passaging_series.csv")
    rows = []
    for (panel, d), g in ser.groupby(["panel", "direction"]):
        u = g[g.arm == "untreated"].sort_values("pdl")
        c = g[g.arm == "chronic radiation"].sort_values("pdl")
        residual = c.excess.to_numpy() - np.interp(c.pdl, u.pdl, u.excess)
        rows.append(dict(panel=panel, direction=d, n=len(c),
                         raw_shift=float(np.mean(c.excess.to_numpy() - u.excess.to_numpy())),
                         pdl_adjusted_shift=float(residual.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(common.SCORES / "chronic_radiation_pdl_adjusted.csv", index=False)
    return out


def main():
    tests = split_panel(pd.read_csv(common.SCORES / "contrast_tests.csv")).dropna(subset=["loss_jeffreys"])
    per = split_panel(pd.read_csv(common.SCORES / "per_sample_scores.csv")).dropna(subset=["loss_frac"])
    parts = ("loss_frac", "loss_frac_raw", "loss_frac_bg")
    j = tests.set_index(KEY + ["direction"]).loss_jeffreys
    per["unit"] = per.trt.fillna(per["sample"])
    per["J"] = j.reindex(pd.MultiIndex.from_frame(per[KEY + ["direction"]])).to_numpy()
    per = per.dropna(subset=["J"])

    rows, units = [], []
    for k, g in per.groupby(KEY, dropna=False):
        grp = g.drop_duplicates("unit").set_index("unit").group
        w = g.set_index(["unit", "direction"])[[*parts, "J"]].unstack("direction")
        dirs = [d for d in ("low", "high") if ("J", d) in w.columns]
        den = sum(w[("J", d)].fillna(0) for d in dirs)
        comb = {c: sum(w[(c, d)].fillna(0) * w[("J", d)].fillna(0) for d in dirs) / den for c in parts}
        w = pd.DataFrame(comb).assign(group=grp).rename_axis("unit").reset_index()
        units.append(pd.DataFrame(dict(zip(KEY, k)) | dict(
            unit=w.unit.to_numpy(), group=w.group.to_numpy(), frac_lost=w.loss_frac.to_numpy(),
            frac_lost_raw=w.loss_frac_raw.to_numpy(), frac_lost_bg=w.loss_frac_bg.to_numpy())))

        t = tests[np.logical_and.reduce([tests[c].isna() if pd.isna(v) else tests[c] == v
                                         for c, v in zip(KEY, k)])]
        paired = (t.design == "paired").all()
        a, c = (w, None) if paired else (w[w.group == "trt"], w[w.group == "ctrl"])
        testable = len(a) >= 5 if paired else min(len(a), len(c)) > 1

        def test_of(col):
            """Per-unit contrast values of one combined column, and its p."""
            if paired:
                v = a[col].to_numpy()
                return v, (stats.wilcoxon(v).pvalue if testable else np.nan)
            v = (a[col] - c[col].mean()).to_numpy()
            return v, (stats.mannwhitneyu(a[col], c[col], alternative="two-sided").pvalue if testable else np.nan)

        v, p = test_of("loss_frac")
        v_raw, p_raw = test_of("loss_frac_raw")
        v_bg, _ = test_of("loss_frac_bg")
        if paired:
            test, design = ("Wilcoxon signed-rank" if testable else "none (n<5)"), "paired"
            lo, hi = bootstrap_ci(v)
            mde = mde80(v)
        else:
            test, design = ("Mann-Whitney" if testable else "descriptive"), "unpaired"
            lo, hi = bootstrap_ci(a.loss_frac.to_numpy(), c.loss_frac.to_numpy())
            mde = mde80(a.loss_frac.to_numpy(), c.loss_frac.to_numpy())

        jt, n_probes = float(t.loss_jeffreys.sum()), int(t.loss_n_probes.sum())
        half = t.set_index("direction")
        rows.append(dict(zip(KEY, k)) | dict(
            design=design, n=len(v) if paired else f"{len(a)} vs {len(c)}", n_probes=n_probes, jeffreys_nats=jt,
            frac_lost=float(np.mean(v)), frac_lost_lo=lo, frac_lost_hi=hi, mde80=float(mde),
            bits_per_cpg=float(np.mean(v)) * jt / (n_probes * np.log(2)),
            frac_lost_raw=float(np.mean(v_raw)), p_raw=float(p_raw), frac_lost_bg=float(np.mean(v_bg)),
            frac_lost_alt=float((t.loss_frac_alt * t.loss_jeffreys).sum() / jt),
            excess_low=float(t[t.direction == "low"].excess_shift.mean()),
            excess_high=float(t[t.direction == "high"].excess_shift.mean()),
            **{f"{col}_{d}": float(half[src].get(d, np.nan)) for d in ("low", "high")
               for col, src in (("frac_lost", "loss_frac"), ("p", "loss_p"))},
            jeffreys_share_low=float(half.loss_jeffreys.get("low", 0.0)) / jt,
            test=test, p=float(p)))

    out = pd.DataFrame(rows).sort_values(["panel_base", "layer_split", "dataset", "contrast"])
    out.to_csv(common.SCORES / "information_loss_summary.csv", index=False)
    pd.concat(units, ignore_index=True).to_csv(common.SCORES / "per_sample_information_loss.csv", index=False)

    adj = chronic_radiation_vs_passaging()

    pd.set_option("display.width", 250)
    print("\nChronic radiation, before and after adjusting for population doublings:")
    print(adj.round(4).to_string(index=False))
    fib = out[(out.panel_base == "fibroblast") & out.age.isna()]
    for layer in ("identity", "history"):
        print(f"\nFibroblast panel, {layer} CpGs - fraction of identity information lost:")
        g = fib[fib.layer_split == layer]
        print(g[["dataset", "contrast", "n", "frac_lost", "frac_lost_lo", "frac_lost_hi", "p",
                 "frac_lost_raw", "frac_lost_bg", "mde80"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
