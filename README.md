# Loss of cell-identity DNA methylation tracks division history, not DNA damage

Code, result tables and figures for the manuscript of the same title (Sagy, Benedek & Bar;
citation to be added on publication). Everything here is reproducible from public GEO data
with `./run_all.sh` (about 20 minutes).

## What is measured

A cell type's **identity CpGs** are those where it sits clearly apart from every other
primary cell type — below all of them (UL) or above all of them (UH). If DNA damage
erodes epigenetic information, these are the CpGs that should erode.

Each perturbation is scored two ways. The **excess shift** is how far the panel moves
relative to a background of non-panel CpGs matched on baseline methylation, which is what
makes a shift interpretable: panel CpGs sit at extreme beta by construction, and extreme
beta regresses on its own. The **information loss** is the same movement expressed as a
fraction of the identity information the panel carries, from a Bernoulli cross-entropy
between the panel's own cell type and a generic other cell (`src/step2_entropy_weights.py`).
Zero is unchanged, one means every CpG moved the whole way to the generic cell's value,
negative is divergence away from it. It is an information-weighted displacement toward the
generic cell at the panel's CpGs, not a measure of everything that makes a cell a fibroblast. In the tables the excess shift has the opposite sign (`excess_shift`, + = away from
the other cell types); the manuscript and figures report it negated, so that on both scales
positive means toward the other cell types, i.e. identity loss. It is net of the same matched background: from each contrast we subtract the loss
the panel would have shown had every CpG moved like its own baseline-matched background
CpGs (the background shifts weighted by the panel CpG's own weight). The unadjusted value
and the background term are kept alongside (`loss_frac_raw`, `loss_frac_bg`).

Panels are also split by whether 30 continuously proliferating cell lines share the state.
CpGs none of them shares are **identity** CpGs; those a fifth or more of them share are
**proliferation-history** CpGs. These are operational labels, and the two layers are always
reported separately.

## Result

Fraction of the fibroblast panel's identity information lost beyond matched background
(+ = toward the other cell types, − = away from them):

| Contrast | n | Identity | History | p (identity) |
|---|---:|---:|---:|---:|
| Replicative exhaustion (GSE197723) | 24 | **+0.243** | −0.496 | 1.2e-7 |
| Replicative senescence (GSE227160, DSF2 / DSF3) | 2 / 2 | +0.110 / +0.150 | −0.107 / −0.212 | — |
| Progeria, HGPS vs control (GSE149960) | 9 v 6 | +0.091 | +0.189 | 0.69 |
| RAS oncogene-induced senescence (GSE197723) | 24 | +0.054 | −0.114 | 1.2e-7 |
| **X-ray (GSE197723)** | 24 | **−0.005** | +0.013 | 0.008 |
| Chronic low-dose radiation (GSE197723) | 4 | −0.021 | −0.009 | — |
| Doxorubicin-induced senescence (GSE227160, DSF2 / DSF3) | 2 / 2 | +0.022 / −0.008 | −0.013 / +0.007 | — |
| Irradiation-induced senescence (GSE227160, DSF2 / DSF3) | 2 / 2 | −0.010 / −0.015 | +0.015 / +0.013 | — |
| Irradiated clones, donor 1 (GSE112812) | 35 | −0.111 (−0.048 excl. 2 outlier controls) | −0.164 | 0.002 |
| Irradiated clones, donor 2, not fibroblast-like (GSE112873) | 26 | +0.009 | −0.018 | 0.025 |

Acute X-irradiation costs no identity information beyond matched background (−0.005,
95% CI −0.007 to −0.002), in a design that would detect a loss of 0.004 on this score with
80% power, about one sixtieth of replicative exhaustion's. Without the background
subtraction the same arm reads as a +0.010 loss (p = 5e-6), all of it change shared with
CpGs at the same baseline methylation. Progeria moves the proliferation-history layer
(p = 0.002), not detectably the identity layer; against the four age-matched controls alone
the two layers move 0.210 (p = 0.006) and 0.026 (p = 0.94). Donor 2's control clones
correlate best with renal epithelium rather than with fibroblasts, and its score stays close
to zero.

Supplementary only. Skin (GSE51954), genotype-paired sun-exposed vs sun-protected biopsies,
all ages, net of matched background:

| Panel | Layer | n pairs | Identity | History | p (identity) |
|---|---|---:|---:|---:|---:|
| Keratinocyte | epidermis | 18 | **+0.069** | +0.029 | <1e-4 |
| Fibroblast | dermis | 19 | +0.038 | +0.011 | 0.011 |
| Keratinocyte | dermis | 19 | −0.036 | −0.013 | 0.002 |
| Fibroblast | epidermis | 18 | +0.004 | −0.032 | 0.42 (n.s.) |

Keratinocyte panels in dermis read as carry-over between layers: sun-exposed dermis holds
more keratinocyte signal. In epidermis the keratinocyte proliferation-history layer moves
less than half as far as the identity layer (p = 0.17), so the two layers neither dissociate
as they do in culture nor move equally. The manuscript reports this arm without drawing a
conclusion from it, because one cohort cannot separate ultraviolet damage from the
inflammation, hyperproliferation and clonal replacement that accompany it.

## Reproducing

```bash
pip install -r requirements.txt
# download the data (see below), then
./run_all.sh
```

Steps 1–9 rebuild every table in `results/` and every figure in `figures/` from the raw
data. They are deterministic: background CpG draws are seeded, so a rerun reproduces the
tables exactly.

| Step | Script | What it does | Time |
|---:|---|---|---:|
| 0 | `step0_prepare_data.py` | GSE40699 IDATs → beta table (one-off, separate environment, see below) | — |
| 1 | `step1_panels.py` | identity panels per cell type, split into identity / history CpGs | ~1 min |
| 2 | `step2_entropy_weights.py` | per-CpG cross-entropy weights q_T, q_O, Δlogit, Jeffreys | ~2 min |
| 3 | `step3_cohorts.py` | clone phenotypes, donor identity, GSE51954 genotype pairing, GSE197723 donor genotypes | ~2 min |
| 4 | `step4_scores.py` | every contrast scored per sample, both ways | ~5 min |
| 5 | `step5_probe_level.py` | the skin result decomposed per CpG | ~1 min |
| 6 | `step6_summary.py` | UL and UH combined into one number per contrast | <1 min |
| 7 | `step7_figures.py` | the five main and five supplementary figures (runs after step 8) | <1 min |
| 8 | `step8_layer_split_sensitivity.py` | does the identity / history split depend on the proliferating lines? | ~2 min |
| 9 | `step9_sensitivity_checks.py` | background pool without any panel CpG, progeria age-matched controls, GSM3084195, chip-level X-ray | ~6 min |

`src/common.py` holds the paths and dataset loaders; it computes nothing.

## Data

Put each dataset in `data/external/<accession>/`. `data/` may be a symlink. Nothing in
`data/` is tracked by git.

| Accession | Files to download | Used for |
|---|---|---|
| `GSE40699` | `GSE40699_RAW.tar` (IDATs) → `idats_clean/`, and the series matrix | ENCODE primary cell types; defines the panels |
| `GSE197723` | series matrix + `GSE197723_datBetaNormalized.csv.gz` | passaging, X-ray, RAS, chronic low-dose radiation; keratinocyte and HUVEC references |
| `GSE227160` | series matrix | replicative vs damage-induced senescence, 2 donors |
| `GSE149960` | series matrix | HGPS progeria vs control dermal fibroblasts |
| `GSE112812`, `GSE112873` | series matrix + supplementary beta file | irradiated single-cell-derived fibroblast clones |
| `GSE51954` | series matrix | sun-exposed vs sun-protected epidermis and dermis |

Five of the six are used straight from their GEO series matrix. GSE40699 has no beta table
in GEO — only raw IDATs — so step 0 processes them once with `methylprep` (sample sheet
derived from the series matrix; defaults, i.e. infer channel switch, quality mask, NOOB).
methylprep 1.7.1 calls `DataFrame.append`, removed in pandas 2.0, so run that one step in
its own environment (steps 1 to 9 are unaffected):

```bash
python -m venv .venv-methylprep
.venv-methylprep/bin/pip install 'methylprep==1.7.1' 'pandas<2'
.venv-methylprep/bin/python src/step0_prepare_data.py
```

## Outputs

```
results/panels/    panel_<celltype>.csv        one row per panel CpG: direction, cancer_side_frac, layer
                   layer_split_sensitivity.csv does the identity / history split survive other constructions?
                   entropy_weights_<celltype>.csv.gz   q_T, q_O, Δlogit, Jeffreys for every CpG
                   build_summary.csv           panel sizes and independent-batch replication (GSE197723
                                               fibroblasts meeting the panel rule again), with the rate
                                               expected by chance and binomial / Fisher p
results/cohorts/   clone_phenotypes.csv        irradiated clones: donor, group, karyotype stability
                   clone_donor_identity.csv    each donor's clones vs every ENCODE cell type
                   gse51954_genotype_pairs.csv recovered sun-exposed / sun-protected pairs
                   gse197723_donor_genotypes.csv  per quadruplet: genotype discordance with its own
                                               members and with the nearest other quadruplet
results/scores/    contrast_tests.csv          one row per contrast x panel x direction — the main table
                                               (loss_frac net of background; loss_frac_raw, loss_frac_bg)
                   per_sample_scores.csv       the per-sample values behind it
                   information_loss_summary.csv  UL+UH combined - the tables above, with bootstrap CI
                                               and mde80 (minimum detectable loss at 80% power);
                                               frac_lost_low/_high, p_low/_high are the two halves
                   per_sample_information_loss.csv   the per-sample values behind it
                   skin_probe_level_*.csv      per-CpG decomposition of the skin result
                   frac_bins.csv               shift vs fraction of proliferating lines sharing the state
                   passaging_series.csv        chronic-radiation series against population doublings
                   chronic_radiation_pdl_adjusted.csv  that arm vs the same passage (raw_shift) and
                                               at matched doublings (pdl_adjusted_shift)
                   skin_validation.csv         panel specificity check in sun-protected skin
                   figure_conditions_summary.csv       the numbers as drawn on Supplementary Figure S1A
                   sensitivity_background_pool.csv     the fibroblast results with every cell-type-panel
                                               CpG excluded from the background pool
                   sensitivity_checks.csv      progeria vs age-matched controls only, history minus
                                               identity, GSM3084195 excluded or as a control, X-ray
                                               averaged to one value per chip
```

## Figures

| File | Manuscript | Content |
|---|---|---|
| `figure_1_concept` | Figure 1 | what a fibroblast-unique CpG is, what identity loss would look like, which designs separate damage from division |
| `figure_2_two_layers` | Figure 2 | the whole panel moves away from other cell types under exhaustion; split into identity and proliferation-history CpGs |
| `figure_3_information_loss` | Figure 3 | the 24 matched cultures as identity information lost; the bound on the X-ray result; the background term |
| `figure_4_all_perturbations` | Figure 4 | every contrast on that scale; two more donors; chronic radiation; progeria |
| `figure_5_irradiated_clones` | Figure 5 | irradiated clones that kept dividing, with two comparisons |
| `figure_S1_unsplit_panel` | Figure S1 | every contrast on the whole, unsplit panel, in Δβ |
| `figure_S2_clones_unsplit` | Figure S2 | the raw data behind Figure 5: clone by clone, whole panel |
| `figure_S3_ul_uh_background` | Figure S3 | UL and UH halves and the background term, per contrast |
| `figure_S4_skin_celltype_panels` | Figure S4 | sun-exposed vs sun-protected skin, per person |
| `figure_S5_skin_probe_level` | Figure S5 | the same contrast with the CpG as the unit |

Every figure uses one sign (positive = toward the other cell types) and one colour code
(identity CpGs blue, proliferation-history CpGs amber, background or unsplit panel grey).

The skin arm is supplementary on purpose: one cohort cannot separate ultraviolet damage from
the inflammation, hyperproliferation and clonal replacement that come with it, so no conclusion
in the manuscript rests on it.

Figures read effect sizes and p-values from `results/`, so a figure cannot disagree with a
table. Two comparisons are computed in the figure script from the per-sample tables: unstable
vs stable clones, and the skin carry-over regression.
