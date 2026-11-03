"""Does the identity / proliferation-history split depend on how the proliferating lines are used?

step1 separates the two layers with a one-bit label: does any of 30 continuously proliferating
GSE40699 lines sit beyond the fibroblast threshold at this CpG? That label is the only place
transformed cells enter the analysis, so it is worth asking two questions about it.

1. Would putting the same 30 lines among the COMPARATORS instead, so that proliferation CpGs
   never enter the panel at all, change any conclusion? (It does not, at the cost of nearly all
   UH CpGs.)
2. Can any rule that uses no proliferating-line data isolate the identity layer? Three candidates
   are tried: how far outside the comparator range the fibroblast value sits, how tightly the
   comparators themselves cluster, and whether the unique call reproduces in an independent batch
   of fibroblasts, i.e. whether GSE197723 fibroblasts also meet the panel rule against the ENCODE
   comparators. (None of them does.)

Each subset is scored the same way the paper scores everything else: mean excess over
baseline-matched background across the 24 matched GSE197723 quadruplets.

Writes results/panels/layer_split_sensitivity.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import step1_panels as P

ARMS = (("Replication", 1), ("X-ray", 2), ("RAS", 3))


def arm_shifts(beta, quads, ref, panel, cgs):
    """Mean excess shift per arm over the quadruplets, UL and UH pooled by CpG count."""
    sub = panel.loc[panel.index.intersection(pd.Index(cgs))]
    out = {"n_UL": int((sub.direction == "low").sum()), "n_UH": int((sub.direction == "high").sum())}
    per_arm = {a: [] for a, _ in ARMS}
    for d in ("low", "high"):
        pr = [c for c in sub.index[sub.direction == d] if c in beta.index and c in ref.index]
        if len(pr) < 5:
            continue
        bg = common.matched_background(ref, pd.Index(pr))
        sgn = common.SIGN[d]
        for arm, idx in ARMS:
            diff = [sgn * ((beta.loc[pr, q[idx]].mean() - beta.loc[bg, q[idx]].mean())
                           - (beta.loc[pr, q[0]].mean() - beta.loc[bg, q[0]].mean())) for q in quads]
            per_arm[arm].append((len(pr), float(np.mean(diff))))
    for arm, vals in per_arm.items():
        out[arm] = sum(w * v for w, v in vals) / sum(w for w, _ in vals) if vals else np.nan
    return out


def main():
    enc = common.load_encode_beta()
    others = enc[P.PRIMARY_NONFIB].dropna(how="all")
    target = enc[P.FIB_SKIN].mean(axis=1)
    panel = pd.read_csv(common.PANELS / "panel_fibroblast_skin.csv").set_index("cg")

    quads = common.FIB_QUADS
    huvec_n = common.gse197723_huvec()
    need = sorted({n for q in quads for n in q} | set(common.KER_REF) | set(common.FIB_REF) | set(huvec_n))
    g197, _ = common.load_gse197723(need)
    ref = g197[[q[0] for q in quads]].mean(axis=1).dropna()
    fib197 = g197[common.FIB_REF].mean(axis=1)

    # ---- primary-cell-only features, computed per panel CpG ------------------
    low = panel.direction.eq("low").to_numpy()
    margin = np.where(low, panel.p5_others - panel.target_mean, panel.target_mean - panel.p95_others)
    o = others.reindex(panel.index)
    iqr = (o.quantile(0.75, axis=1) - o.quantile(0.25, axis=1)).to_numpy()
    f = fib197.reindex(panel.index)
    # the fibroblast-unique rule, met again by GSE197723 fibroblasts against the ENCODE comparators
    reps = np.where(low, f <= panel.p5_others - P.GATE, f >= panel.p95_others + P.GATE)
    tight = iqr <= np.nanquantile(iqr, 0.2)

    rows = []

    is_id = panel.layer.eq("identity")

    def add(rule, how, cgs):
        sel = panel.index.isin(pd.Index(cgs))
        rows.append(dict(rule=rule, uses_proliferating_lines=how,
                         **arm_shifts(g197, quads, ref, panel, cgs),
                         identity_recall=float((sel & is_id).sum() / is_id.sum()),
                         identity_precision=float((sel & is_id).sum() / max(sel.sum(), 1))))

    idx = panel.index
    add("identity layer (the paper's panel)", "label", idx[panel.layer == "identity"])
    add("proliferation-history layer", "label", idx[panel.layer == "history"])
    add("whole panel, no split", "no", idx)
    add("margin outside comparator range, top 5%", "no", idx[margin >= np.nanquantile(margin, 0.95)])
    add("margin outside comparator range, top 10%", "no", idx[margin >= np.nanquantile(margin, 0.90)])
    add("comparators tightly clustered, IQR bottom 20%", "no", idx[tight])
    add("unique call reproduces in an independent batch", "no", idx[reps])
    add("reproduces AND comparators tight", "no", idx[reps & tight])

    # ---- the same 30 lines used as comparators instead of as a label ---------
    alt = P.build(target, enc[P.PRIMARY_NONFIB + P.CANCER_IMMORTAL].dropna(how="all")).set_index("cg")
    shared = len(set(alt.index) & set(panel.index[panel.layer == "identity"]))
    rows.append(dict(rule="panel built with the 30 lines among the comparators",
                     uses_proliferating_lines="comparator",
                     **arm_shifts(g197, quads, ref, alt, alt.index)))
    rows[-1]["identity_cpgs_shared"] = shared

    out = pd.DataFrame(rows)
    out.to_csv(common.PANELS / "layer_split_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
