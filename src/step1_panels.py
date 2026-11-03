"""Build the cell-type identity panels, and split them into identity and history CpGs.

A CpG belongs to a cell type's panel if that cell type sits clearly outside every
comparator cell type: UL if target <= 5th percentile(others) - 0.1, UH if
target >= 95th percentile(others) + 0.1, one value per cell type (replicates averaged).

Comparators are the 20 primary non-fibroblast GSE40699 types. They are primary on
purpose: continuously proliferating lines hypermethylate Polycomb-target CpG islands, which
raises the 95th percentile there and removes nearly every UH candidate. Fibroblast lines
are excluded from the comparators of the fibroblast panels, so the panel measures
fibroblast identity rather than skin-fibroblast-vs-other-fibroblast.

Keeping proliferating lines out of the comparators lets in CpGs where the cultured target
shares its state with them - PMD-like loss at UL, Polycomb gain at UH - which marks a CpG as
a plausible record of proliferation rather than identity. Each CpG therefore also gets

    cancer_side_frac = fraction of the 30 continuously proliferating lines (tumour lines,
                       HEK293, MCF10A and lymphoblastoid lines) also beyond the target's threshold

    layer = identity (frac == 0) | history (frac >= 0.2) | mixed (in between)

and every result downstream is reported per layer. The mixed CpGs are kept in the panel
files but belong to neither layer.

Independent-batch replication (fibroblast panels): the share of a layer's CpGs at which
GSE197723 fibroblasts also meet the panel rule (0.1 beyond the 5th or 95th percentile of the
ENCODE comparators), against the same test at every array CpG outside the panel (chance),
with a binomial p; identity and history layers are compared with Fisher's exact test. CpGs are treated as independent, which they are not
quite, so these p-values are approximate.

Writes results/panels/panel_<name>.csv and build_summary.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common

GATE = 0.1
HISTORY_FRAC = 0.2

FIB_SKIN = ["BJ", "AG04449", "AG09309", "AG10803", "NHDF-neo"]  # foreskin, fetal thigh, toe, abdomen, dermis
FIB_LUNG = ["IMR90", "AG04450"]                                 # fetal lung - the comparison panel for the clones
FIB_ALL_NORMAL = FIB_SKIN + FIB_LUNG + ["HCF", "AG09319"]       # + cardiac, gum: the fibroblast comparator centroid
PRIMARY_NONFIB = ["AoSMC", "HAEpiC", "HCM", "HCPEpiC", "HEEpiC", "HIPEpiC", "HMEC", "HNPCEpiC", "HPAEpiC",
                  "HRCEpiC", "HRE", "HRPEpiC", "HUVEC", "Hepatocytes", "NH-A", "NHBE", "PrEC", "RPTEC",
                  "SAEC", "SKMC"]
CANCER_IMMORTAL = ["A549", "BE2_C", "CMK", "Caco-2", "ECC-1", "HCT-116", "HeLa-S3", "HepG2", "HL-60", "Jurkat",
                   "K562", "LNCaP", "MCF-7", "NB4", "PANC-1", "PFSK-1", "SK-N-MC", "SK-N-SH", "SK-N-SH_RA", "T-47D",
                   "U87", "ovcar-3", "NT2-D1", "HEK293", "MCF10A-Er-Src", "GM06990", "GM12878", "GM12891",
                   "GM12892", "GM19239"]


def build(target, others, same_batch=None, replicate=None):
    """target: one Series. others: probes x comparator cell types."""
    p5, p95 = others.quantile(0.05, axis=1), others.quantile(0.95, axis=1)
    ul = target <= p5 - GATE
    uh = target >= p95 + GATE
    if same_batch is not None:          # must also clear comparators run in the target's own batch
        ul &= target <= same_batch.min(axis=1) - GATE
        uh &= target >= same_batch.max(axis=1) + GATE
    if replicate is not None:           # must also hold in an independent batch of the target
        ul &= replicate <= p5 - GATE
        uh &= replicate >= p95 + GATE
    out = pd.DataFrame({"cg": target.index, "target_mean": target.values,
                        "p5_others": p5.values, "p95_others": p95.values})
    out["direction"] = None
    out.loc[ul.fillna(False).values, "direction"] = "low"
    out.loc[uh.fillna(False).values, "direction"] = "high"
    return out.dropna(subset=["direction"]).reset_index(drop=True)


def add_layer(panel, cancer):
    """cancer: probes x continuously proliferating lines."""
    c = cancer.reindex(panel.cg)
    low = panel.direction.eq("low").to_numpy()
    beyond_low = c.le(panel.p5_others.to_numpy() - GATE, axis=0).to_numpy()
    beyond_high = c.ge(panel.p95_others.to_numpy() + GATE, axis=0).to_numpy()
    frac = np.where(low[:, None], beyond_low, beyond_high).mean(axis=1)
    panel["cancer_side_frac"] = frac
    panel["layer"] = np.select([frac == 0, frac >= HISTORY_FRAC], ["identity", "history"], "mixed")
    return panel


def replication_chance(panel, fib, p5, p95):
    """The replication test applied to every array CpG outside the panel: the rate expected by chance."""
    bg = fib.notna() & p5.notna() & ~fib.index.isin(panel.cg)
    hit = {"low": fib <= p5 - GATE, "high": fib >= p95 + GATE}
    return {d: float(v[bg].mean()) for d, v in hit.items()}


def gse197723_references(enc):
    """Keratinocyte and endothelial targets and comparators, which need the second batch.

    Keratinocytes are absent from GSE40699, so their reference comes from GSE197723.
    Fibroblasts and HUVEC were profiled in both batches; each enters the comparators as one
    centroid averaged over the two, so that every cell type counts once, as it does everywhere
    else. Returns probes on both platforms only.
    """
    huvec_n = common.gse197723_huvec()
    g197, _ = common.load_gse197723(common.KER_REF + common.FIB_REF + huvec_n)
    shared = enc.index.intersection(g197.index)
    e, g = enc.loc[shared], g197.loc[shared]
    ref = dict(ker=g[common.KER_REF].mean(axis=1), fib197=g[common.FIB_REF].mean(axis=1),
               huv197=g[huvec_n].mean(axis=1), huvec_encode=e["HUVEC"])
    fib = pd.concat([e[FIB_ALL_NORMAL].mean(axis=1), ref["fib197"]], axis=1).mean(axis=1).rename("fibroblast")
    huvec = pd.concat([e["HUVEC"], ref["huv197"]], axis=1).mean(axis=1).rename("HUVEC")
    nonfib = e[[t for t in PRIMARY_NONFIB if t != "HUVEC"]]
    ref["others_k"] = pd.concat([nonfib, fib, huvec], axis=1)
    ref["others_e"] = pd.concat([nonfib, fib, ref["ker"].rename("keratinocyte")], axis=1)
    return ref


def main():
    enc = common.load_encode_beta()
    missing = [t for t in FIB_ALL_NORMAL + PRIMARY_NONFIB + CANCER_IMMORTAL if t not in enc.columns]
    assert not missing, missing
    others = enc[PRIMARY_NONFIB]

    panels, notes = {}, {}
    for name, members in (("fibroblast_skin", FIB_SKIN), ("fibroblast_lung", FIB_LUNG)):
        panels[name] = build(enc[members].mean(axis=1), others.dropna(how="all"))
        notes[name] = f"target {'+'.join(members)}; comparators {len(PRIMARY_NONFIB)} primary non-fibroblast types"

    # The same-batch gate keeps batch differences from manufacturing keratinocyte uniqueness.
    # HUVEC is in GSE40699, and must replicate in the GSE197723 HUVEC of the other batch.
    r = gse197723_references(enc)
    ker, fib197, huv197 = r["ker"], r["fib197"], r["huv197"]
    panels["keratinocyte"] = build(ker, r["others_k"], same_batch=pd.concat([fib197, huv197], axis=1))
    notes["keratinocyte"] = (f"GSE197723 keratinocytes vs {r['others_k'].shape[1]} types (19 primary GSE40699 + "
                             f"fibroblast + HUVEC, each averaged over both batches); must also clear same-batch "
                             f"fibroblast/HUVEC")
    panels["endothelial"] = build(r["huvec_encode"], r["others_e"], replicate=huv197)
    notes["endothelial"] = (f"ENCODE HUVEC vs {r['others_e'].shape[1]} types (19 primary GSE40699 + fibroblast "
                            f"averaged over both batches + keratinocyte); must replicate in GSE197723 HUVEC")

    cancer = enc[CANCER_IMMORTAL]
    o = others.reindex(fib197.index)
    p5_all, p95_all = o.quantile(0.05, axis=1), o.quantile(0.95, axis=1)
    rows = []
    for name, panel in panels.items():
        panel = add_layer(panel, cancer)
        panel.to_csv(common.PANELS / f"panel_{name}.csv", index=False)
        if name.startswith("fibroblast"):
            chance = replication_chance(panel, fib197, p5_all, p95_all)
        for (d, layer), grp in panel.groupby(["direction", "layer"]):
            row = dict(panel=name, direction=d, layer=layer, n=len(grp), construction=notes[name])
            if name.startswith("fibroblast"):
                # independent batch: do GSE197723 fibroblasts meet the same rule against the ENCODE comparators?
                ids = [c for c in grp.cg if c in fib197.index]
                f, g = fib197[ids], grp.set_index("cg").loc[ids]
                ok = (f <= g.p5_others - GATE) if d == "low" else (f >= g.p95_others + GATE)
                row.update(replicates_gse197723=float(ok.mean()), n_tested=len(ids), n_replicated=int(ok.sum()),
                           replicates_chance=chance[d],
                           replicates_p_vs_chance=stats.binomtest(int(ok.sum()), len(ids), chance[d]).pvalue)
            rows.append(row)
    summ = pd.DataFrame(rows)
    # identity vs history layer, per fibroblast panel and direction (Fisher's exact test on CpG counts)
    for _, g in summ[summ.n_tested.notna()].groupby(["panel", "direction"]):
        a, b = g[g.layer == "identity"], g[g.layer == "history"]
        table = [[r.n_replicated.iloc[0], r.n_tested.iloc[0] - r.n_replicated.iloc[0]] for r in (a, b)]
        summ.loc[a.index.union(b.index), "replicates_p_identity_vs_history"] = stats.fisher_exact(table).pvalue
    summ.to_csv(common.PANELS / "build_summary.csv", index=False)
    print(summ.drop(columns="construction").to_string(index=False))


if __name__ == "__main__":
    main()
