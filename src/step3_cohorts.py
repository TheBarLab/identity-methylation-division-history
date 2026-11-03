"""Sample bookkeeping that the scoring and figure steps need: who is paired with whom.

Four things:

1. Irradiated-clone phenotypes (GSE112812/GSE112873) and, for each donor, how the
   untreated clones correlate with each GSE40699 cell type. Donor 1's clones are
   fibroblast-like; donor 2's are not, which is why donor 2 acts as a null - there is no
   fibroblast identity there to lose.

2. Raw mean panel beta per donor-1 clone, for the irradiation figure.

3. Genotype pairing of GSE51954. The sun-exposed and sun-protected biopsies come from the
   same people but GEO does not say which pairs with which, and the title indices are
   unreliable in young epidermis. Pairs are recovered from the data: correlate samples on
   SNP-like trimodal probes and solve the assignment problem within each layer and age
   group. Pairs with r < 0.9 are dropped downstream.

4. Donor structure of the GSE197723 quadruplets. GEO labels every fibroblast "age 0, male"
   and has no donor field, and the quadruplets are read from chip position. Genotype calls
   at probes that are discrete (within 0.08 of 0, 0.5 or 1) in every sample and polymorphic
   among the series' 17 skin donors show whether each quadruplet is one person and whether
   the quadruplets are different people.

Writes results/cohorts/ and results/scores/irradiation_raw_panel_beta.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import common

N_TOP_VARIABLE = 20000


def donor_identity(beta, ph, enc):
    """Correlation of each donor's untreated clones with every ENCODE cell type."""
    shared = beta.index.intersection(enc.index)
    top = enc.loc[shared].var(axis=1).nlargest(N_TOP_VARIABLE).index
    rows = []
    for donor in common.CLONE_SERIES:
        ctrl = ph[(ph.donor == donor) & (ph.group == "non-irradiated")].sample_col
        r = enc.loc[top].corrwith(beta.loc[top, ctrl].mean(axis=1)).sort_values(ascending=False)
        rows += [dict(donor=donor, rank=rank, cell_line=line, r=val,
                      is_fibroblast=line in common.FIBROBLAST_LINES)
                 for rank, (line, val) in enumerate(r.items(), 1)]
    return pd.DataFrame(rows)


def raw_panel_beta(beta, ph, panel):
    """Mean beta over the unsplit fibroblast panel, per donor-1 clone."""
    sub = ph[ph.donor == "donor1"]
    groups = {"control": sub[sub.group == "non-irradiated"].sample_col.tolist(),
              "stable": sub[(sub.group == "irradiated") & (sub.stability == "stable")].sample_col.tolist(),
              "unstable": sub[(sub.group == "irradiated") & (sub.stability == "unstable")].sample_col.tolist()}
    irr = sub[sub.group == "irradiated"].sample_col.tolist()
    b = beta[groups["control"] + irr].dropna()
    rows = []
    for d in ("low", "high"):
        probes = [p for p in panel[panel.direction == d].cg if p in b.index]
        m = b.loc[probes].mean()
        for g, cols in groups.items():
            rows += [dict(direction=d, group=g, sample=s, n_probes=len(probes), panel_mean_beta=float(m[s]))
                     for s in cols]
    return pd.DataFrame(rows)


def pair_by_genotype(b, m):
    """Match sun-exposed to sun-protected samples of the same person, per layer and age group."""
    bb = b.dropna()
    mid = ((bb > 0.35) & (bb < 0.65)).mean(1)
    lo, hi = (bb < 0.15).mean(1), (bb > 0.85).mean(1)
    snp = bb.index[(mid > 0.15) & ((lo > 0.15) | (hi > 0.15)) & (mid + lo + hi > 0.9)]
    pairs = []
    for (grp, layer), g in m.groupby(["grp", "tissue"]):
        e, p = g[g.exposure == "sun exposed"], g[g.exposure == "sun protected"]
        C = np.corrcoef(bb.loc[snp, e.gsm].T.values, bb.loc[snp, p.gsm].T.values)[:len(e), len(e):]
        r, c = linear_sum_assignment(-C)
        for i, j in zip(r, c):
            pairs.append(dict(grp=grp, layer=layer, exposed=e.gsm.iloc[i], protected=p.gsm.iloc[j],
                              r=float(C[i, j]), title_idx_match=e.idx.iloc[i] == p.idx.iloc[j]))
    return pd.DataFrame(pairs), len(snp)


def gse197723_donors():
    """Per quadruplet: genotype discordance with its own members and with the nearest other quadruplet."""
    ph = common.gse197723_pheno().set_index("n")
    skin = ph[ph["cell type"] == "Skin"].index.tolist()
    quads = common.FIB_QUADS
    b, _ = common.load_gse197723(skin + [n for q in quads for n in q])
    b = b.dropna()
    near = np.minimum.reduce([b.abs(), (b - 0.5).abs(), (b - 1).abs()]) < 0.08
    calls = pd.DataFrame(np.select([b < 0.25, b > 0.75], [0, 2], 1), index=b.index, columns=b.columns)
    calls = calls[near.all(axis=1) & (calls[skin].nunique(axis=1) > 1)]
    ctrl = [q[0] for q in quads]
    disc = pd.DataFrame({a: [(calls[a] != calls[c]).mean() for c in ctrl] for a in ctrl}, index=ctrl)
    rows = []
    for q in quads:
        other = disc[q[0]].drop(q[0])
        rows.append(dict(control=q[0], members=" ".join(map(str, q[1:])), chip=ph.loc[q[0], "description"],
                         max_discordance_own_members=max((calls[q[0]] != calls[m]).mean() for m in q[1:]),
                         nearest_other_control=int(other.idxmin()), discordance_nearest_other=float(other.min())))
    return pd.DataFrame(rows), len(calls)


def main():
    print("GSE197723 quadruplets: donor structure...")
    donors, n_snp = gse197723_donors()
    donors.to_csv(common.COHORTS / "gse197723_donor_genotypes.csv", index=False)
    same = donors[donors.discordance_nearest_other < 0.01]
    print(f"  {n_snp} genotype probes; own-member discordance <= {donors.max_discordance_own_members.max():.3f}; "
          f"{len(donors) - len(same) // 2} distinct donors among {len(donors)} quadruplets "
          f"(sharing a genotype: {same.control.tolist()})")

    print("Irradiated clones (GSE112812 / GSE112873)...")
    beta, ph = common.load_clone_beta()
    ph = ph.rename(columns={"strain": "donor"})
    ph.to_csv(common.COHORTS / "clone_phenotypes.csv", index=False)
    print("  " + "; ".join(f"{d}: {len(g)} clones" for d, g in ph.groupby("donor")))

    ident = donor_identity(beta, ph, common.load_encode_beta())
    ident.to_csv(common.COHORTS / "clone_donor_identity.csv", index=False)
    for donor, g in ident.groupby("donor"):
        best_fib = g[g.is_fibroblast].iloc[0]
        print(f"  {donor}: closest cell type {g.iloc[0].cell_line} (r={g.iloc[0].r:.3f}); "
              f"closest fibroblast {best_fib.cell_line} (r={best_fib.r:.3f}, rank {best_fib['rank']})")

    panel = pd.read_csv(common.PANELS / "panel_fibroblast_skin.csv")
    raw = raw_panel_beta(beta, ph, panel)
    raw.to_csv(common.SCORES / "irradiation_raw_panel_beta.csv", index=False)
    del beta

    print("Skin (GSE51954)...")
    b51, m51 = common.read_series_matrix(common.EXT / "GSE51954" / "GSE51954_series_matrix.txt.gz")
    m51["gsm"] = m51.geo_accession
    m51["grp"] = m51.title.str.extract(r"from (young|old)")
    m51["idx"] = m51.title.str.extract(r"(\d+)$").astype(int)
    pairs, n_snp = pair_by_genotype(b51, m51)
    pairs.to_csv(common.COHORTS / "gse51954_genotype_pairs.csv", index=False)
    keep = pairs[pairs.r >= 0.9]
    print(f"  {n_snp} SNP-like probes; kept {len(keep)}/{len(pairs)} pairs at r>=0.9 "
          f"({int(keep.title_idx_match.sum())} of which the title index also matches)")
    print(keep.groupby("layer").size().to_string())


if __name__ == "__main__":
    main()
