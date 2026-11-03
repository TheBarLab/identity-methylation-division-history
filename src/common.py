"""Paths, dataset loaders and the matched-background draw shared by every step.

Nothing here computes a result; it only reads the six GEO datasets into a common
shape (probes x samples, beta values) so the analysis steps can stay short.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
PANELS = ROOT / "results" / "panels"
COHORTS = ROOT / "results" / "cohorts"
SCORES = ROOT / "results" / "scores"
FIGURES = ROOT / "figures"

for _d in (PANELS, COHORTS, SCORES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# A UL CpG sits below every comparator cell type, a UH CpG above. The sign makes "+"
# mean the same thing for both: moving away from the panel's cell type.
SIGN = {"low": -1.0, "high": 1.0}

# Background draw: K CpGs per panel CpG, matched on baseline beta in BIN-wide bins.
K_BACKGROUND = 20
BIN = 0.02


def read_series_matrix(path):
    """A GEO series matrix -> (beta: probes x GSM, metadata: one row per GSM)."""
    meta, rows, header = {}, [], None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_"):
                p = [x.strip('"') for x in line.rstrip("\n").split("\t")]
                key, vals = p[0][8:], p[1:]
                if key == "characteristics_ch1":
                    key = vals[0].split(":")[0].strip()
                    vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
                while key in meta:
                    key += "_"
                meta[key] = vals
            elif line.startswith('"ID_REF"'):
                header = [x.strip('"') for x in line.rstrip("\n").split("\t")]
            elif header and line.startswith('"cg'):
                rows.append(line.rstrip("\n").split("\t"))
    beta = pd.DataFrame(rows, columns=header).set_index("ID_REF")
    beta.index = beta.index.str.strip('"')
    beta = beta.apply(pd.to_numeric, errors="coerce").astype("float32")
    return beta, pd.DataFrame(meta)


def matched_background(ref, probes, seed=1, k=K_BACKGROUND, exclude=None):
    """For each scored CpG, k other CpGs from the same bin of baseline beta `ref`.

    The pool is every CpG except `probes`, i.e. except the panel layer and direction being
    scored, so CpGs from other layers of the same panel stay eligible. `exclude` removes
    further CpGs from the pool (step9 excludes every cell-type-panel CpG as a sensitivity check).
    Returns the multiset with duplicates kept, so that a mean over it weights each panel
    CpG's bin equally. Deterministic given `seed`.
    """
    bins = (ref // BIN).astype(int)
    pool = bins.drop(probes)
    if exclude is not None:
        pool = pool.drop(pool.index.intersection(list(exclude)))
    by_bin = {b: ix.index.to_numpy() for b, ix in pool.groupby(pool)}
    rng = np.random.default_rng(seed)
    bg = []
    for p in probes:
        cand = by_bin.get(bins[p])
        if cand is None or len(cand) == 0:            # empty bin: fall back to the nearest one
            cand = pool.index[(pool - bins[p]).abs() == (pool - bins[p]).abs().min()].to_numpy()
        bg.append(rng.choice(cand, size=k))
    return np.concatenate(bg)


# --------------------------------------------------------------- GSE40699 (ENCODE cell types)
ENCODE_DIR = EXT / "GSE40699" / "idats_clean"
FIBROBLAST_LINES = {"IMR90", "ProgFib", "BJ", "AG04449", "AG04450",
                    "AG09309", "AG09319", "AG10803", "HCF", "NHDF-neo"}


def load_encode_beta():
    """GSE40699 betas, one column per cell type (replicates averaged). See step0_prepare_data.py."""
    beta = pd.read_pickle(ENCODE_DIR / "beta_values.pkl")
    ss = pd.read_csv(ENCODE_DIR / "samplesheet.csv")
    # Column names look like "hg19_wgEncodeHaibMethyl450Ag04449SitesRep1"; the sample sheet's
    # Sentrix_Position field carries the same string, which is what maps a column to a cell type.
    rename = {}
    for col in beta.columns:
        hit = ss.loc[ss["Sentrix_Position"].apply(lambda s: str(s) in col), "Sample_Name"]
        if len(hit) == 1:
            rename[col] = hit.iloc[0]
    if len(rename) != len(beta.columns):
        raise RuntimeError(f"mapped only {len(rename)}/{len(beta.columns)} ENCODE columns to cell types")
    return beta.rename(columns=rename).T.groupby(level=0).mean().T


# --------------------------------------------------------------- GSE197723 (perturbed fibroblasts)
# Sample numbers within GSE197723, from its own titles ("Sample<N>...").
KER_REF = [13, 14, 15, 16, 78, 83, 86]             # untreated keratinocytes (reference)
KER_CHRONIC = [(99, 102), (100, 103), (101, 104)]  # untreated vs chronic radiation, matched on doublings
FIB_REF = [153 + 4 * k for k in range(24)]         # the untreated control of each quadruplet
# Each quadruplet is (control, replicative exhaustion, X-ray, RAS) from one donor culture.
FIB_QUADS = [(153 + 4 * k, 154 + 4 * k, 155 + 4 * k, 156 + 4 * k) for k in range(24)]
FIB_CHRONIC = [(106, 110), (107, 111), (108, 112), (109, 113)]
# Population doublings of the chronic-radiation passaging series (untreated 106-109, chronic 110-113).
FIB_CHRONIC_PDL = {106: 3.2, 107: 23.9, 108: 43.9, 109: 54.1, 110: 2.9, 111: 21.8, 112: 39.3, 113: 49.4}


def gse197723_pheno():
    """Sample metadata, indexed by the sample number in the title ("Sample<N>...")."""
    _, ph = read_series_matrix(EXT / "GSE197723" / "GSE197723_series_matrix.txt.gz")
    ph["n"] = ph.title.str.extract(r"Sample(\d+)").astype(int)
    return ph


def load_gse197723(sample_numbers):
    """Only the requested samples; the full beta table is ~2.4 GB compressed."""
    ph = gse197723_pheno()
    pos = ph.set_index("n").description
    path = EXT / "GSE197723" / "GSE197723_datBetaNormalized.csv.gz"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    want = {pos[n]: n for n in sample_numbers}
    cols = [c for c in header[1:] if c in want]
    beta = pd.read_csv(path, usecols=[header[0]] + cols, index_col=0, dtype={c: "float32" for c in cols})
    beta.columns = [want[c] for c in beta.columns]
    return beta, ph.set_index("n")


def gse197723_huvec():
    ph = gse197723_pheno()
    return ph[(ph["cell type"] == "HUVEC") & (ph.condition == "None")].n.tolist()


# --------------------------------------------------------------- GSE112812 / GSE112873 (clones)
# Single-cell-derived fibroblast clones from two donors, irradiated or not.
CLONE_SERIES = {"donor1": EXT / "GSE112812" / "GSE112812_series_matrix.txt.gz",
                "donor2": EXT / "GSE112873" / "GSE112873_series_matrix.txt.gz"}
# The series matrices hold metadata only; betas are in the supplementary files, whose columns are
# "SAMPLE N" = the Nth GSM in series-matrix order (the missing N are the starting-culture samples).
CLONE_BETA = {"donor1": EXT / "GSE112812" / "GSE112812_Normalized_AverageBeta_pro_control_irradiated.txt.gz",
              "donor2": EXT / "GSE112873" / "GSE112873_M_pro_control-irradiated.txt.gz"}
# Two donor-1 control clones whose panel scores are far outside the rest; every donor-1 result is
# reported with and without them.
CLONE_OUTLIER_CONTROLS = ["GSM3084161", "GSM3084174"]


def parse_clone_phenotypes(path, donor):
    rows: dict[str, list[str]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                rows["gsm"] = [s.strip('"') for s in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                rows["title"] = [s.strip('"') for s in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [s.strip('"') for s in line.split("\t")[1:]]
                if not vals or ":" not in vals[0]:
                    continue
                key = vals[0].split(":", 1)[0].strip().replace(" ", "_").lower()
                rows[key] = [v.split(":", 1)[1].strip() if ":" in v else "" for v in vals]
    pheno = pd.DataFrame(rows)
    pheno["strain"] = donor
    pheno["sample_col"] = [f"{donor}_sample_{i + 1}" for i in range(len(pheno))]
    return pheno


def load_clone_beta():
    """Both donors -> (beta on shared probes, phenotypes without the starting cultures)."""
    frames, phs = [], []
    for donor, path in CLONE_SERIES.items():
        ph = parse_clone_phenotypes(path, donor)
        d = pd.read_csv(CLONE_BETA[donor], sep="\t", index_col=0)
        pos = {i + 1: c for i, c in enumerate(ph["sample_col"])}
        d.columns = [pos[int(c.replace("SAMPLE", "").strip())] for c in d.columns]
        frames.append(d.astype("float32"))
        phs.append(ph)
    beta = pd.concat(frames, axis=1, join="inner")
    ph = pd.concat(phs, ignore_index=True)
    return beta, ph[ph["group"] != "starting culture"].reset_index(drop=True)
