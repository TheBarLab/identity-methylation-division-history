"""Turn the GSE40699 IDATs into the beta table every later step reads.

Five of the six datasets are used straight from their GEO series matrix or supplementary
beta file. GSE40699 (ENCODE primary cell types), which defines the panels, has no beta
table in GEO at all — only raw IDATs — so it has to be processed once:

    data/external/GSE40699/idats_clean/{beta_values.pkl, samplesheet.csv}

The sample sheet is written from the series matrix (Sample_Name is the GEO
source_name_ch1, i.e. the cell-type label the panels are keyed on), then methylprep runs
with its defaults (sesame=True: infer channel switch, quality mask, NOOB + dye bias).

methylprep 1.7.1 calls DataFrame.append, which pandas removed in 2.0, so this one step
needs an environment with pandas < 2 (the analysis steps do not). See README.
"""
from __future__ import annotations

import sys

import pandas as pd

import common

IDATS = common.ENCODE_DIR


def write_samplesheet():
    _, m = common.read_series_matrix(common.EXT / "GSE40699" / "GSE40699_series_matrix.txt.gz")
    supp = [c for c in m.columns if c.startswith("supplementary_file")]
    rows = []
    for _, r in m.iterrows():
        # ".../GSM999377_hg19_wgEncodeHaibMethyl450T47dDm002p24hSitesRep1_Grn.idat.gz"
        # One row per array, not per GSM: HCT-116 (GSM999387) was run twice.
        for c in supp:
            name = r[c].rsplit("/", 1)[-1]
            if not name.endswith("_Grn.idat.gz"):
                continue
            stem = name.split("_")
            rows.append(dict(GSM_ID=r.geo_accession, Sample_Name=r.source_name_ch1,
                             Sentrix_ID=stem[1], Sentrix_Position=stem[2]))
    ss = pd.DataFrame(rows)
    path = IDATS / "samplesheet.csv"
    ss.to_csv(path, index=False)
    print(f"{path}: {len(ss)} samples, {ss.Sample_Name.nunique()} cell types")
    return path


def main():
    if not IDATS.exists() or not list(IDATS.glob("*.idat.gz")):
        sys.exit(f"Put the GSE40699 IDATs in {IDATS} first (see README, Data).")
    sheet = write_samplesheet()

    if pd.__version__ >= "2":
        sys.exit("methylprep 1.7.1 needs pandas < 2 (it calls DataFrame.append).\n"
                 "Run this step in a separate environment, e.g.\n"
                 "  python -m venv .venv-methylprep && .venv-methylprep/bin/pip install "
                 "'methylprep==1.7.1' 'pandas<2' && .venv-methylprep/bin/python src/step0_prepare_data.py\n"
                 "then use your normal environment for steps 1-7.")

    import methylprep
    methylprep.run_pipeline(str(IDATS), betas=True, sample_sheet_filepath=str(sheet))
    beta = pd.read_pickle(IDATS / "beta_values.pkl")
    print(f"beta_values.pkl: {beta.shape[0]:,} probes x {beta.shape[1]} arrays")


if __name__ == "__main__":
    main()
