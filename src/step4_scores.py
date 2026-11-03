"""Score every contrast, two ways, at the level of samples rather than CpGs.

Each sample gets two scores on each panel:

    excess  = sign * (mean beta over panel CpGs - mean beta over matched background CpGs)
              + means divergence away from the panel's cell type.

    loss    = information lost beyond the matched background, as a fraction of the panel's
              total identity information (step2_entropy_weights.py). + means collapse
              towards a generic cell, i.e. the opposite sign to excess.

The background matters: panel CpGs sit at extreme beta by construction, and extreme beta
regresses. Every panel CpG is therefore matched to 20 non-panel CpGs in the same 0.02-wide
bin of control-mean beta. The excess subtracts their mean beta. The loss subtracts the loss
the panel would have shown had each of its CpGs moved like its own 20 background CpGs, i.e.
the background shifts weighted by the panel CpG's dlogit. Both columns that go into that
difference are kept: loss_frac_raw (the panel's shift alone) and loss_frac_bg (the
background term), with loss_frac = loss_frac_raw - loss_frac_bg. Both scores are computed
on the same complete-case CpGs and the same background draw.

Both are compared between conditions at the sample level - paired Wilcoxon where the
design is paired, Mann-Whitney otherwise - so sample-to-sample variability is part of the
test. A CpG-level permutation would not do that. GSE227160 has one young culture per
donor and stays descriptive.

Writes results/scores/{contrast_tests,per_sample_scores,passaging_series,frac_bins,
skin_validation}.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common

LN2 = float(np.log(2))
WEIGHT_FILE = {"fibroblast": "fibroblast_skin", "fibroblast_lung": "fibroblast_lung",
               "keratinocyte": "keratinocyte", "endothelial": "endothelial"}
_WEIGHTS: dict[str, pd.DataFrame] = {}

ROWS: list[dict] = []
PER: list[dict] = []
EXTRA: list[dict] = []


# ------------------------------------------------------------------------------- scores
def weights_for(panel):
    name = WEIGHT_FILE.get(panel.split(":")[0])
    if name is None:
        return None
    if name not in _WEIGHTS:
        w = pd.read_csv(common.PANELS / f"entropy_weights_{name}.csv.gz")
        _WEIGHTS[name] = w.set_index("cg")[["dlogit", "dlogit_alt", "jeffreys"]]
    return _WEIGHTS[name]


def score(beta, ref_cols, probes, direction, panel, seed=1):
    """Per-sample scores of one panel direction, all on the same CpGs and background draw.

    beta should hold only the samples of one contrast. CpGs with a missing value in any of
    them are dropped first. The loss scores are relative to the mean of ref_cols, which
    cancels in a contrast; they are left out if fewer than 5 panel CpGs carry weights.
    """
    beta = beta.dropna()
    ref = beta[ref_cols].mean(axis=1)
    probes = [p for p in probes if p in ref.index]
    if not probes:
        return None
    bg = common.matched_background(ref, probes, seed=seed).reshape(len(probes), -1)
    out = dict(n_probes=len(probes),
               excess=common.SIGN[direction] * (beta.loc[probes].mean() - beta.loc[bg.ravel()].mean()))
    w = weights_for(panel) if panel else None
    keep = pd.Index(probes).isin(w.index) if w is not None else None
    if w is None or keep.sum() < 5:
        return out
    pr = np.asarray(probes)[keep]
    ww = w.loc[pr]
    j = float(ww.jeffreys.sum())
    d_panel = beta.loc[pr].to_numpy() - ref[pr].to_numpy()[:, None]
    bgk = bg[keep]
    d_bg = (beta.loc[bgk.ravel()].to_numpy() - ref[bgk.ravel()].to_numpy()[:, None])
    d_bg = d_bg.reshape(*bgk.shape, beta.shape[1]).mean(axis=1)   # each panel CpG's background shift

    def frac(d, col):
        return pd.Series((d * ww[col].to_numpy()[:, None]).sum(axis=0) / j, index=beta.columns)

    out.update(n_loss=len(pr), jeffreys=j, raw=frac(d_panel, "dlogit"), bg=frac(d_bg, "dlogit"),
               loss_alt=frac(d_panel - d_bg, "dlogit_alt"))
    out["loss"] = out["raw"] - out["bg"]
    return out


def contrast(x, pairs=None, ctrl=None, trt=None, test=True):
    """Per-unit contrast values of a per-sample score, and the two-sided p (nan if untested)."""
    if pairs is not None:
        v = np.array([x[t] - x[c] for c, t in pairs])
        return v, (stats.wilcoxon(v).pvalue if test and len(v) >= 5 else np.nan)
    v = x[trt].to_numpy() - x[ctrl].mean()
    ok = test and len(trt) > 1 and len(ctrl) > 1
    return v, (stats.mannwhitneyu(x[trt], x[ctrl], alternative="two-sided").pvalue if ok else np.nan)


# ------------------------------------------------------------------------------- designs
def record(dataset, contrast_name, panel, d, s, pairs=None, ctrl=None, trt=None, test=True, extra=None):
    """Test one scored panel direction; append its contrast row and its per-sample rows."""
    if s is None:
        return
    design = dict(pairs=pairs) if pairs is not None else dict(ctrl=ctrl, trt=trt)
    v, p = contrast(s["excess"], test=test, **design)
    if pairs is not None:
        n, label = len(v), ("Wilcoxon signed-rank" if test and len(v) >= 5 else "none (n<5)")
    else:
        n, label = f"{len(trt)} vs {len(ctrl)}", ("Mann-Whitney" if not np.isnan(p) else "descriptive")
    row = dict(dataset=dataset, contrast=contrast_name, panel=panel, direction=d, n_probes=s["n_probes"],
               design="paired" if pairs is not None else "unpaired", n=n, excess_shift=float(v.mean()),
               n_positive=int((v > 0).sum()), test=label, p=float(p))
    if "loss" in s:
        lv, lp = contrast(s["loss"], test=test, **design)
        rv, rp = contrast(s["raw"], test=test, **design)
        row.update(loss_frac=float(lv.mean()), loss_p=float(lp), loss_n_probes=s["n_loss"],
                   loss_jeffreys=s["jeffreys"], loss_bits_per_cpg=float(lv.mean()) * s["jeffreys"] / (s["n_loss"] * LN2),
                   loss_frac_alt=float(contrast(s["loss_alt"], test=False, **design)[0].mean()),
                   loss_frac_raw=float(rv.mean()), loss_p_raw=float(rp),
                   loss_frac_bg=float(contrast(s["bg"], test=False, **design)[0].mean()))
    ROWS.append(row | (extra or {}))

    cols = ("loss", "raw", "bg")
    names = ("loss_frac", "loss_frac_raw", "loss_frac_bg")
    base = dict(dataset=dataset, contrast=contrast_name, panel=panel, direction=d)
    if pairs is not None:
        for (c, t), ev in zip(pairs, v):
            PER.append(base | dict(ctrl=c, trt=t, excess_diff=float(ev))
                       | {nm: (float(s[k][t] - s[k][c]) if k in s else np.nan) for k, nm in zip(cols, names)}
                       | (extra or {}))
    else:
        y = s["excess"][ctrl].mean()
        for smp in trt + ctrl:
            PER.append(base | dict(group="trt" if smp in trt else "ctrl", sample=smp,
                                   excess_rel_ctrl=float(s["excess"][smp] - y))
                       | {nm: (float(s[k][smp]) if k in s else np.nan) for k, nm in zip(cols, names)})


def paired(dataset, contrast_name, beta, pairs, probes, d, panel="fibroblast", extra=None):
    ctrl = [c for c, _ in pairs]
    s = score(beta[ctrl + [t for _, t in pairs]], ctrl, probes, d, panel)
    record(dataset, contrast_name, panel, d, s, pairs=pairs, extra=extra)
    return s


def unpaired(dataset, contrast_name, beta, ctrl, trt, probes, d, panel="fibroblast", test=True):
    s = score(beta[ctrl + trt], ctrl, probes, d, panel)
    record(dataset, contrast_name, panel, d, s, ctrl=ctrl, trt=trt, test=test)


def probes_for(panel_df, d):
    return panel_df[panel_df.direction == d].cg.tolist()


def panel_sets():
    """Each panel both unsplit (":all") and split by CpG layer (see step1_panels.py)."""
    def split(name, label):
        p = pd.read_csv(common.PANELS / f"panel_{name}.csv")
        return {f"{label}:all": p, **{f"{label}:{layer}": g for layer, g in p.groupby("layer")}}

    fib = split("fibroblast_skin", "fibroblast")
    skin = {**fib, **split("keratinocyte", "keratinocyte"), **split("endothelial", "endothelial")}
    return fib, split("fibroblast_lung", "fibroblast_lung"), skin


# ------------------------------------------------------------------------------- datasets
def irradiated_clones(fib_sets, lung_sets):
    """GSE112812/GSE112873: clones that were irradiated and kept dividing."""
    beta, ph = common.load_clone_beta()
    ph = ph.rename(columns={"strain": "donor"})
    d1 = ph[ph.donor == "donor1"]
    irr = d1[d1.group == "irradiated"].sample_col.tolist()
    for label, ctrl in [
            ("irradiated clones (donor 1)", d1[d1.group == "non-irradiated"].sample_col.tolist()),
            ("irradiated clones (donor 1), excl. 2 outlier controls",
             d1[(d1.group == "non-irradiated") & ~d1.gsm.isin(common.CLONE_OUTLIER_CONTROLS)].sample_col.tolist())]:
        b = beta[ctrl + irr]
        for lab, pn in {**fib_sets, **lung_sets}.items():
            for d in ("low", "high"):
                unpaired("GSE112812", label, b, ctrl, irr, probes_for(pn, d), d, panel=lab)
    d2 = ph[ph.donor == "donor2"]
    ctrl2 = d2[d2.group == "non-irradiated"].sample_col.tolist()
    irr2 = d2[d2.group == "irradiated"].sample_col.tolist()
    for lab, pn in fib_sets.items():
        for d in ("low", "high"):
            unpaired("GSE112873", "irradiated clones (donor 2, not fibroblast-like)",
                     beta[ctrl2 + irr2], ctrl2, irr2, probes_for(pn, d), d, panel=lab)


def perturbed_fibroblasts(fib_sets, skin_sets):
    """GSE197723: matched quadruplets per culture, plus the chronic-radiation passaging series."""
    quads, chronic = common.FIB_QUADS, common.FIB_CHRONIC
    load = {n for q in quads for n in q} | {n for p in chronic for n in p} | \
           {n for p in common.KER_CHRONIC for n in p}
    b, _ = common.load_gse197723(sorted(load))

    for lab, pn in fib_sets.items():
        for arm, idx in [("Replication", 1), ("X-ray", 2), ("RAS", 3)]:
            pairs = [(q[0], q[idx]) for q in quads]
            for d in ("low", "high"):
                paired("GSE197723", f"{arm} vs control", b, pairs, probes_for(pn, d), d, panel=lab)
        for d in ("low", "high"):
            s = paired("GSE197723", "chronic radiation vs untreated", b, chronic, probes_for(pn, d), d, panel=lab)
            for smp, v in s["excess"][list(common.FIB_CHRONIC_PDL)].items():
                EXTRA.append(dict(table="passaging_series", panel=lab, direction=d, sample=smp,
                                  arm="untreated" if smp < 110 else "chronic radiation",
                                  pdl=common.FIB_CHRONIC_PDL[smp], excess=float(v)))

    # Identity and history are the ends of a continuum: bin the panel by the fraction of
    # continuously proliferating lines sharing the state and watch the passaging shift rise with it.
    p = pd.read_csv(common.PANELS / "panel_fibroblast_skin.csv")
    p["bin"] = pd.cut(p.cancer_side_frac, [-0.01, 0, 0.1, 0.2, 0.4, 0.7, 1.0],
                      labels=["0", "0-0.1", "0.1-0.2", "0.2-0.4", "0.4-0.7", "0.7-1"])
    for (d, bn), g in p.groupby(["direction", "bin"], observed=True):
        pr = [c for c in g.cg if c in b.index]
        if len(pr) < 5:
            continue
        for arm, idx in [("Replication", 1), ("X-ray", 2), ("RAS", 3)]:
            pairs = [(q[0], q[idx]) for q in quads]
            ctrl = [c for c, _ in pairs]
            s = score(b[ctrl + [t for _, t in pairs]], ctrl, pr, d, panel=None)
            ex, n = s["excess"], s["n_probes"]
            diff = np.array([ex[t] - ex[c] for c, t in pairs])
            raw = common.SIGN[d] * np.mean([b.loc[pr, t].mean() - b.loc[pr, c].mean() for c, t in pairs])
            EXTRA.append(dict(table="frac_bins", direction=d, bin=bn, n_probes=n, arm=arm,
                              excess_shift=float(diff.mean()), raw_signed_shift=float(raw),
                              n_positive=int((diff > 0).sum())))

    for lab, pn in skin_sets.items():
        if lab.startswith("keratinocyte"):
            for d in ("low", "high"):
                paired("GSE197723", "keratinocyte chronic radiation vs untreated", b,
                       common.KER_CHRONIC, probes_for(pn, d), d, panel=lab)


def senescence_by_trigger(fib_sets):
    """GSE227160: replicative senescence vs doxorubicin- and irradiation-induced senescence."""
    b, m = common.read_series_matrix(common.EXT / "GSE227160" / "GSE227160_series_matrix.txt.gz")
    m["gsm"] = m.geo_accession
    fib = m[m.donor.isin(["DSF2", "DSF3"])].copy()
    fib["arm"] = fib.title.str.split("_", n=2).str[2]
    for donor, g in fib.groupby("donor"):
        young = g[g.arm == "Young"].gsm.tolist()
        for arm in ["Middle_passaged", "RS", "IRR_SIPS", "Dox_SIPS"]:
            trt = g[g.arm.str.startswith(arm)].gsm.tolist()
            for lab, pn in fib_sets.items():
                for d in ("low", "high"):
                    unpaired("GSE227160", f"{donor}: {arm} vs young", b, young, trt,
                             probes_for(pn, d), d, panel=lab, test=False)


def progeria(fib_sets):
    """GSE149960: HGPS vs control dermal fibroblasts (unrelated individuals)."""
    b, m = common.read_series_matrix(common.EXT / "GSE149960" / "GSE149960_series_matrix.txt.gz")
    hgps = m[m["disease state"].str.contains("HGPS")].geo_accession.tolist()
    ctrl = m[m["disease state"] == "normal"].geo_accession.tolist()
    for lab, pn in fib_sets.items():
        for d in ("low", "high"):
            unpaired("GSE149960", "HGPS vs control", b, ctrl, hgps, probes_for(pn, d), d, panel=lab)


def sun_exposed_skin(skin_sets):
    """GSE51954: sun-exposed vs sun-protected epidermis and dermis of the same people."""
    b, m = common.read_series_matrix(common.EXT / "GSE51954" / "GSE51954_series_matrix.txt.gz")
    pairs = pd.read_csv(common.COHORTS / "gse51954_genotype_pairs.csv")
    pairs = pairs[pairs.r >= 0.9]
    protected = m[m.exposure == "sun protected"]
    for k, pn in skin_sets.items():
        for d in ("low", "high"):
            pr = probes_for(pn, d)
            if len(pr) < 5:
                continue
            for layer, g in pairs.groupby("layer"):
                # One background draw per layer, against all of its sun-protected samples, so
                # the age groups are slices of the same scores and average to the "all" row.
                prot = g.protected.tolist()
                s = score(b[prot + g.exposed.tolist()], prot, pr, d, k)
                for age, gg in list(g.groupby("grp")) + [("all", g)]:
                    record("GSE51954", f"sun exposed vs protected, {layer}", k, d, s,
                           pairs=list(zip(gg.protected, gg.exposed)), extra=dict(layer=layer, age=age))
            # Sanity check that each panel is where it should be: a keratinocyte panel should
            # read high in epidermis and low in dermis even without any sun exposure.
            present = [p for p in pr if p in b.index]
            for layer in ("dermis", "epidermis"):
                cols = protected[protected.tissue == layer].geo_accession
                EXTRA.append(dict(table="skin_validation", panel=k, direction=d, layer=layer,
                                  n_probes=len(present),
                                  mean_beta=float(b.loc[present, cols].mean().mean())))


def main():
    fib_sets, lung_sets, skin_sets = panel_sets()
    print("GSE112812 / GSE112873: irradiated clones...")
    irradiated_clones(fib_sets, lung_sets)
    print("GSE197723: replication, X-ray, RAS, chronic radiation...")
    perturbed_fibroblasts(fib_sets, skin_sets)
    print("GSE227160: senescence by trigger...")
    senescence_by_trigger(fib_sets)
    print("GSE149960: progeria...")
    progeria(fib_sets)
    print("GSE51954: sun-exposed skin...")
    sun_exposed_skin(skin_sets)

    res = pd.DataFrame(ROWS)
    res.to_csv(common.SCORES / "contrast_tests.csv", index=False)
    pd.DataFrame(PER).to_csv(common.SCORES / "per_sample_scores.csv", index=False)
    for t, g in pd.DataFrame(EXTRA).groupby("table"):
        g.drop(columns="table").dropna(axis=1, how="all").to_csv(common.SCORES / f"{t}.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    show = res[(res.panel == "fibroblast:identity") & (res.dataset != "GSE51954")]
    print(show[["dataset", "contrast", "direction", "n", "excess_shift", "p", "loss_frac", "loss_frac_raw",
                "loss_frac_bg"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
