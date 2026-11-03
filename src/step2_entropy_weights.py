"""Per-CpG weights that turn a methylation shift into bits of lost cell identity.

Treat each CpG on each molecule as Bernoulli and describe a sample under two codes: the
panel's own cell type, q_T, and a generic other cell, q_O (the mean of the comparator
primary types). The extra code length of the identity model over the generic one is

    dH_i = H(b_i, q_T) - H(b_i, q_O) = const_i - b_i * (logit q_T - logit q_O),

linear in beta, so the constant drops out of any contrast and leaves

    information lost (nats) = sum_i (b_i^trt - b_i^ctrl) * dlogit_i,
    dlogit_i = logit q_O - logit q_T.

Moving a CpG all the way from q_T to q_O costs its Jeffreys term,

    J_i = (q_O - q_T) * (logit q_O - logit q_T)  >= 0,

so dividing the sum by sum_i J_i gives the fraction of the panel's identity information
that was lost: 0 = unchanged, 1 = every CpG moved the whole way to the generic cell's value,
< 0 = divergence away from it. Because dlogit carries the sign, UL and UH are two halves of one quantity
rather than two directions needing a sign convention.

q is clipped away from 0 and 1 before the logit; dlogit_alt repeats it at a tighter clip
so the sensitivity to that choice can be reported.

References are built exactly as in step1_panels.py, but for every CpG rather than only
panel CpGs, so the matched background can be scored on the same scale.

Writes results/panels/entropy_weights_<name>.csv.gz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import step1_panels as P

CLIP, CLIP_ALT = 0.02, 0.01


def logit(q, clip):
    q = np.clip(q, clip, 1 - clip)
    return np.log(q / (1 - q))


def weights(q_target, others):
    """others: probes x comparator cell types."""
    q_o = others.mean(axis=1)
    df = pd.DataFrame({"cg": q_target.index, "q_target": q_target.values, "q_others": q_o.values,
                       "q_others_median": others.median(axis=1).values}).dropna()
    for name, c in (("dlogit", CLIP), ("dlogit_alt", CLIP_ALT)):
        df[name] = logit(df.q_others, c) - logit(df.q_target, c)
    df["jeffreys"] = (df.q_others - df.q_target) * df.dlogit
    return df


def main():
    enc = common.load_encode_beta()
    others = enc[P.PRIMARY_NONFIB].dropna(how="all")
    out = {name: weights(enc[m].mean(axis=1), others)
           for name, m in (("fibroblast_skin", P.FIB_SKIN), ("fibroblast_lung", P.FIB_LUNG))}

    r = P.gse197723_references(enc)
    out["keratinocyte"] = weights(r["ker"], r["others_k"])
    out["endothelial"] = weights(r["huvec_encode"], r["others_e"])

    for name, df in out.items():
        df.to_csv(common.PANELS / f"entropy_weights_{name}.csv.gz", index=False, float_format="%.6g")
        panel = pd.read_csv(common.PANELS / f"panel_{name}.csv")
        m = df.set_index("cg").reindex(panel.cg)
        w = m.jeffreys.abs()
        print(f"{name}: {len(df):,} CpGs; panel {len(panel):,}; "
              f"mean |dlogit| panel {m.dlogit.abs().mean():.2f} vs all {df.dlogit.abs().mean():.2f}; "
              f"Jeffreys sum {m.jeffreys.sum():.0f} nats; "
              f"effective n (participation ratio) {w.sum() ** 2 / (w ** 2).sum():.0f}")


if __name__ == "__main__":
    main()
