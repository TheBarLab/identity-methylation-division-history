"""The skin result decomposed per CpG instead of per person.

step4 averages a panel within each person and tests across people. Here the CpG is the
unit: for every panel CpG, the mean sun-exposed minus sun-protected difference across the
genotype-matched pairs of one skin layer, against the same matched background. This says
whether a panel-level shift is the whole panel moving together or a tail of CpGs doing
all the work - which the sample-level test cannot distinguish.

signed_delta = SIGN[direction] * delta_beta, so + is divergence and - is collapse, as
everywhere else. Per-CpG p-values are Wilcoxon across pairs, with Benjamini-Hochberg FDR.

Writes results/scores/skin_probe_level_{deltas,density,summary}.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common

PANEL_FILES = {"fibroblast": "panel_fibroblast_skin.csv", "keratinocyte": "panel_keratinocyte.csv"}
GRID = 201


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    print("Loading GSE51954...")
    b, _ = common.read_series_matrix(common.EXT / "GSE51954" / "GSE51954_series_matrix.txt.gz")
    pairs = pd.read_csv(common.COHORTS / "gse51954_genotype_pairs.csv")
    pairs = pairs[pairs.r >= 0.9]
    panels = {k: pd.read_csv(common.PANELS / f) for k, f in PANEL_FILES.items()}

    deltas, dens, summ = [], [], []
    for layer, g in pairs.groupby("layer"):
        prot, expo = g.protected.tolist(), g.exposed.tolist()
        sub = b[prot + expo].dropna()
        ref = sub[prot].mean(axis=1)
        d_all = pd.DataFrame(sub[expo].to_numpy() - sub[prot].to_numpy(), index=sub.index)  # probes x pairs
        print(f"{layer}: {len(g)} pairs, {len(sub):,} probes with complete data")
        for pname, pdf in panels.items():
            for d in ("low", "high"):
                probes = [c for c in pdf[pdf.direction == d].cg if c in sub.index]
                bg = common.matched_background(ref, probes)
                sgn = common.SIGN[d]
                dp, db = d_all.loc[probes], d_all.loc[bg]
                mp, mb = sgn * dp.mean(axis=1).to_numpy(), sgn * db.mean(axis=1).to_numpy()
                w = stats.wilcoxon(dp.to_numpy(), axis=1).pvalue
                q = bh(w)
                deltas.append(pd.DataFrame(dict(
                    layer=layer, panel=pname, direction=d, cg=probes, n_pairs=len(g),
                    baseline_beta=ref[probes].to_numpy(), delta_beta=dp.mean(axis=1).to_numpy(),
                    signed_delta=mp, p=w, fdr=q)))
                x = np.linspace(min(mp.min(), mb.min()), max(mp.max(), mb.max()), GRID)
                for lab, v in (("panel", mp), ("background", mb)):
                    dens.append(pd.DataFrame(dict(layer=layer, panel=pname, direction=d, set=lab,
                                                  x=x, density=stats.gaussian_kde(v)(x))))
                summ.append(dict(
                    layer=layer, panel=pname, direction=d, n_pairs=len(g), n_probes=len(probes),
                    mean_signed=float(mp.mean()), median_signed=float(np.median(mp)),
                    sd_signed=float(mp.std(ddof=1)), bg_mean_signed=float(mb.mean()),
                    bg_sd_signed=float(mb.std(ddof=1)), excess=float(mp.mean() - mb.mean()),
                    frac_collapse=float((mp < 0).mean()), bg_frac_collapse=float((mb < 0).mean()),
                    n_fdr_collapse=int(((q < 0.05) & (mp < 0)).sum()),
                    n_fdr_diverge=int(((q < 0.05) & (mp > 0)).sum()),
                    mannwhitney_p=float(stats.mannwhitneyu(mp, mb).pvalue)))

    pd.concat(deltas).to_csv(common.SCORES / "skin_probe_level_deltas.csv", index=False)
    pd.concat(dens).to_csv(common.SCORES / "skin_probe_level_density.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(common.SCORES / "skin_probe_level_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
