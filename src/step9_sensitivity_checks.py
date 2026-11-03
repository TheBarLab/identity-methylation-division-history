"""Sensitivity checks quoted in the text, none of which changes a conclusion.

1. Background pool. The matched background is drawn from every CpG outside the panel layer
   and direction being scored, so CpGs from other layers of the same panel are eligible.
   Steps 4 and 6 are rerun into a temporary folder with every CpG of all four cell-type
   panels excluded from the pool, and the fibroblast identity and history rows are compared.
2. Progeria controls. GSE149960's six controls are two parental lines (PRF) and four
   age-matched lines (Coriell). HGPS against the four age-matched controls alone, and the
   within-sample history-minus-identity difference, from the stored per-sample scores.
3. GSM3084195 is irradiated by its group field but titled as a control clone. The clone
   contrasts with that sample excluded, or counted as a control.
4. Array rows. In GSE197723 each condition sits on the same two rows of every chip; the
   X-ray contrast averaged to one value per chip.

Writes results/scores/sensitivity_background_pool.csv and results/scores/sensitivity_checks.csv.
"""
from __future__ import annotations

import functools
import gzip
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import common

PANEL_FILES = ("fibroblast_skin", "fibroblast_lung", "keratinocyte", "endothelial")
KEYS = ["dataset", "contrast", "panel_base", "layer_split", "age"]
AGE_MATCHED = ["GSM4518928", "GSM4518929", "GSM4518930", "GSM4518931"]   # GM05659 GM00969 GM01864 GM02036
MISLABELLED = "donor1_sample_41"                                        # GSM3084195


def background_pool(published: pd.DataFrame) -> pd.DataFrame:
    excl = set()
    for name in PANEL_FILES:
        excl |= set(pd.read_csv(common.PANELS / f"panel_{name}.csv")["cg"])
    original_scores, original_draw = common.SCORES, common.matched_background
    with tempfile.TemporaryDirectory() as tmp:
        common.SCORES = Path(tmp)
        common.matched_background = functools.partial(original_draw, exclude=excl)
        try:
            import step4_scores
            import step6_summary
            step4_scores.ROWS.clear(); step4_scores.PER.clear(); step4_scores.EXTRA.clear()
            step4_scores.main()
            step6_summary.main()
            alt = pd.read_csv(common.SCORES / "information_loss_summary.csv")
        finally:
            common.SCORES, common.matched_background = original_scores, original_draw
    m = published.merge(alt, on=KEYS, suffixes=("", "_excluded"))
    m = m[(m.panel_base == "fibroblast") & m.layer_split.isin(["identity", "history"])]
    cols = ["frac_lost", "frac_lost_lo", "frac_lost_hi", "p"]
    out = m[KEYS + cols + [c + "_excluded" for c in cols]].copy()
    out["n_excluded_cpgs"] = len(excl)
    print(f"  background pool: {len(excl)} panel CpGs excluded; largest change in any fraction lost "
          f"{(published.merge(alt, on=KEYS, suffixes=('', '_x')).eval('abs(frac_lost - frac_lost_x)').max()):.4f}")
    return out


def unpaired_rows(per, dataset, contrast, layer):
    g = per[(per.dataset == dataset) & (per.contrast == contrast) & (per.panel_base == "fibroblast")
            & (per.layer_split == layer)].set_index("unit")
    return g[g.group == "trt"].frac_lost, g[g.group == "ctrl"].frac_lost


def row(check, contrast, layer, t, c):
    return dict(check=check, contrast=contrast, layer=layer, n=f"{len(t)} vs {len(c)}",
                frac_lost=float(t.mean() - c.mean()),
                p=float(stats.mannwhitneyu(t, c, alternative="two-sided").pvalue))


def progeria(per):
    rows, both = [], {}
    for layer in ("identity", "history"):
        t, c = unpaired_rows(per, "GSE149960", "HGPS vs control", layer)
        rows.append(row("progeria, four age-matched controls", "HGPS vs control", layer, t, c.loc[AGE_MATCHED]))
        both[layer] = pd.concat([t, c])
    d = both["history"] - both["identity"]                  # same sample, history minus identity
    trt, ctrl = (x.index for x in unpaired_rows(per, "GSE149960", "HGPS vs control", "identity"))
    rows.append(row("progeria, history minus identity", "HGPS vs control", "history - identity",
                    d.loc[trt], d.loc[ctrl]))
    rows.append(row("progeria, history minus identity, age-matched controls", "HGPS vs control",
                    "history - identity", d.loc[trt], d.loc[AGE_MATCHED]))
    return rows


def mislabelled_clone(per):
    rows = []
    for contrast in ("irradiated clones (donor 1)", "irradiated clones (donor 1), excl. 2 outlier controls"):
        for layer in ("identity", "history"):
            t, c = unpaired_rows(per, "GSE112812", contrast, layer)
            rows.append(row("GSM3084195 excluded", contrast, layer, t.drop(MISLABELLED), c))
            rows.append(row("GSM3084195 as control", contrast, layer, t.drop(MISLABELLED),
                            pd.concat([c, t.loc[[MISLABELLED]]])))
    return rows


def chip_level(per):
    """GSE197723 chip of each sample, from its IDAT file name in the series matrix."""
    chip = {}
    with gzip.open(common.EXT / "GSE197723" / "GSE197723_series_matrix.txt.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_supplementary_file") and "_R0" in line:
                files = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    for t, f in zip(titles, files):
        n, m = re.search(r"Sample\s*(\d+)", t), re.search(r"(\d{12})_R\d\dC\d\d", f)
        if n and m:
            chip[n.group(1)] = m.group(1)
    rows = []
    for arm in ("Replication vs control", "X-ray vs control", "RAS vs control"):
        x = per[(per.dataset == "GSE197723") & (per.contrast == arm) & (per.panel_base == "fibroblast")
                & (per.layer_split == "identity")]
        g = x.assign(chip=x.unit.astype(str).map(chip)).groupby("chip").frac_lost.mean()
        rows.append(dict(check="one value per chip", contrast=arm, layer="identity", n=len(g),
                         frac_lost=float(g.mean()), p=float(stats.wilcoxon(g).pvalue)))
    return rows


def main():
    published = pd.read_csv(common.SCORES / "information_loss_summary.csv")
    per = pd.read_csv(common.SCORES / "per_sample_information_loss.csv")
    print("sensitivity: background pool excluding every cell-type-panel CpG (reruns steps 4 and 6)...")
    background_pool(published).to_csv(common.SCORES / "sensitivity_background_pool.csv", index=False)
    checks = pd.DataFrame(progeria(per) + mislabelled_clone(per) + chip_level(per))
    checks.to_csv(common.SCORES / "sensitivity_checks.csv", index=False)
    print(checks.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
