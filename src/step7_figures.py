"""The manuscript figures, drawn from the tables in results/.

Main
figure_1_concept                 what a fibroblast-unique CpG is, what identity loss would look like,
                                 and which experiments separate DNA damage from division
figure_2_two_layers              the whole panel moves away from other cell types under exhaustion;
                                 split by proliferating-line sharing, the identity CpGs move toward them
figure_3_information_loss        the same 24 cultures as a fraction of identity information; the bound on
                                 the X-ray result; what the background subtraction does
figure_4_all_perturbations       every contrast on that scale; two more donors; chronic radiation; progeria
figure_5_irradiated_clones       clones that kept dividing after 2 Gy, with two comparisons

Supplementary
figure_S1_unsplit_panel          every contrast on the whole, unsplit panel, in delta-beta
figure_S2_clones_unsplit         the raw data behind Figure 5: clone by clone, unsplit panel
figure_S3_ul_uh_background       UL and UH halves and the background term, contrast by contrast
figure_S4_skin_celltype_panels   sun-exposed vs sun-protected skin, per person
figure_S5_skin_probe_level       the same contrast with the CpG as the unit (split violins)

One sign everywhere: positive = toward the other cell types (identity loss), negative = away
from them. The delta-beta score in the tables (excess_shift, + = away) is drawn negated; the
information-lost fraction already has this sign. Colour codes the CpG layer (identity blue,
proliferation history amber, background or unsplit panel grey); marker shape codes UL / UH.

Effect sizes and p-values are read from results/, never recomputed, with two exceptions
computed here from the per-sample tables and marked where they occur: unstable vs stable
clones, and the skin carry-over regression. Figures 1 and 2 also plot raw GSE40699 values
for three example CpGs. Each figure is written as PDF and PNG into figures/.
"""
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Polygon, Ellipse
from scipy import stats

import common

R, PANELS, OUT = common.SCORES, common.PANELS, common.FIGURES

C_ID, C_HIST, C_BG = "#2C5AA0", "#C47A2C", "#9AA3B0"
C_UL, C_UH = "#333333", "#8F8F8F"                       # unsplit panels: UL dark, UH light grey
C_TEXT, C_MUTED, C_GRID = "#1B2A4A", "#7B8494", "#E6E6E1"
C_OTHER = "#BFC4CC"                                      # other primary cell types in the example CpGs
C_CTRL, C_STABLE, C_UNSTABLE = "#7EA8BE", "#E8C547", "#D95E4B"
ARM = {"X-ray": ("Acute X-ray", "#2E9C8A", "^"), "RAS": ("RAS oncogene", "#8A5DB5", "s"),
       "Replication": ("Replicative exhaustion", "#222222", "o")}
ARM_ORDER = ("X-ray", "RAS", "Replication")
DIRS = (("low", C_UL, "o", "UL"), ("high", C_UH, "D", "UH"))
LAYER_MK = {"identity": (C_ID, "o"), "history": (C_HIST, "s")}
FS = dict(tick=12, lab=13, title=14.5, note=11, res=12.5, letter=21, small=10)
ARROWS = "← away from · toward other cell types →"
TOWARD = "Shift beyond background (Δβ)\n" + ARROWS
LOST = "Identity information lost (fraction)\n" + ARROWS

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                     "font.size": 11, "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
                     "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"]})

EX = pd.read_csv(R / "contrast_tests.csv")
PER = pd.read_csv(R / "per_sample_scores.csv")
BINS = pd.read_csv(R / "frac_bins.csv")
SER = pd.read_csv(R / "passaging_series.csv")
VAL = pd.read_csv(R / "skin_validation.csv")
SUMM = pd.read_csv(PANELS / "build_summary.csv")
ILS = pd.read_csv(R / "information_loss_summary.csv")
PLS = pd.read_csv(R / "per_sample_information_loss.csv")
CHRON = pd.read_csv(R / "chronic_radiation_pdl_adjusted.csv")
IDENT = pd.read_csv(common.COHORTS / "clone_donor_identity.csv")
PH = pd.read_csv(common.COHORTS / "clone_phenotypes.csv")
STAB = dict(zip(PH.sample_col, PH.stability))
OUTLIERS = set(PH.sample_col[PH.gsm.isin(common.CLONE_OUTLIER_CONTROLS)])
C1_ALL = "irradiated clones (donor 1)"
C1_EXCL = "irradiated clones (donor 1), excl. 2 outlier controls"
C2 = "irradiated clones (donor 2, not fibroblast-like)"


# ------------------------------------------------------------------------------- helpers
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pfmt(p):
    if p is None or pd.isna(p):
        return "n/a"
    if p == 0:
        return "< 10⁻³⁰⁰"
    if p < 0.001:
        m, e = "{:.1e}".format(p).split("e")
        return "{} × 10{}".format(m, str(int(e)).translate(SUP))
    return "{:.3f}".format(p) if p < 0.05 else "{:.2f}".format(p)


def ptext(p):
    """'p = 0.02', or 'p < 10⁻³⁰⁰' when the p-value underflows."""
    s = pfmt(p)
    return "p " + s if s.startswith("<") else "p = " + s


def clean(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def letter(ax, s, x=-0.1, y=1.08, fs=None):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs or FS["letter"], fontweight="bold", va="top")


def title(ax, s, pad=10, fs=None):
    ax.set_title(s, fontsize=fs or FS["title"], fontweight="bold", color=C_TEXT, loc="left", pad=pad)


def ygrid(ax):
    ax.yaxis.grid(True, color=C_GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=C_GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def strip(ax, x, v, color, marker="o", fill=True, s=30, w=0.3, seed=0, alpha=0.8, lw_mean=2.8):
    v = np.asarray(v, dtype=float)
    jit = np.random.default_rng(seed).uniform(-w / 2.3, w / 2.3, len(v))
    ax.scatter(x + jit, v, s=s, marker=marker, facecolor=color if fill else "white",
               edgecolor="white" if fill else color, linewidths=0.5 if fill else 1.3, alpha=alpha, zorder=3)
    ax.plot([x - w / 2, x + w / 2], [v.mean()] * 2, color=color, lw=lw_mean, zorder=4, solid_capstyle="butt")


def ex_row(dataset, contrast, d, panel, layer=None, age=None):
    q = (EX.dataset == dataset) & (EX.contrast == contrast) & (EX.direction == d) & (EX.panel == panel)
    if layer is not None:
        q &= (EX.layer == layer) & (EX.age == age)
    g = EX[q]
    return g.iloc[0] if len(g) else None


def per_rows(dataset, contrast, d, panel, layer=None, age=None):
    q = (PER.dataset == dataset) & (PER.contrast == contrast) & (PER.direction == d) & (PER.panel == panel)
    if layer is not None:
        q &= PER.layer == layer
        if age is not None:
            q &= PER.age == age
    return PER[q]


def ils_row(dataset, contrast, layer_split, panel_base="fibroblast", layer=None, age=None):
    q = ((ILS.dataset == dataset) & (ILS.contrast == contrast) & (ILS.panel_base == panel_base)
         & (ILS.layer_split == layer_split))
    q &= ILS.layer.isna() if layer is None else ILS.layer == layer
    q &= ILS.age.isna() if age is None else ILS.age == age
    g = ILS[q]
    return g.iloc[0] if len(g) else None


def pls_rows(dataset, contrast, layer_split, panel_base="fibroblast"):
    q = ((PLS.dataset == dataset) & (PLS.contrast == contrast) & (PLS.panel_base == panel_base)
         & (PLS.layer_split == layer_split) & PLS.layer.isna() & PLS.age.isna())
    return PLS[q]


def n_layer(layer, d):
    g = SUMM[(SUMM.panel == "fibroblast_skin") & (SUMM.layer == layer) & (SUMM.direction == d)]
    return int(g.n.iloc[0])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


_EXAMPLES = {}


def example_cpgs():
    """An identity UL, identity UH and history UL CpG where every one of the five fibroblast lines clears
    the threshold, chosen by the widest such clearance; with the GSE40699 values behind them."""
    if _EXAMPLES:
        return _EXAMPLES
    import step1_panels as P1
    panel = pd.read_csv(PANELS / "panel_fibroblast_skin.csv").set_index("cg")
    enc = common.load_encode_beta()
    fib = enc.loc[panel.index, P1.FIB_SKIN]
    low = panel.direction == "low"
    thr = np.where(low, panel.p5_others - P1.GATE, panel.p95_others + P1.GATE)
    panel["clearance"] = np.where(low, thr - fib.max(axis=1), fib.min(axis=1) - thr)
    panel["threshold"] = thr
    sel = {"identity UL": (panel.layer == "identity") & low,
           "identity UH": (panel.layer == "identity") & ~low,
           "history UL": (panel.layer == "history") & low & (panel.cancer_side_frac >= 0.9)}
    for k, m in sel.items():
        row = panel[m].nlargest(1, "clearance").iloc[0]
        _EXAMPLES[k] = dict(row=row, cg=row.name, other=enc.loc[row.name, P1.PRIMARY_NONFIB].astype(float),
                            fib=enc.loc[row.name, P1.FIB_SKIN].astype(float),
                            prol=enc.loc[row.name, P1.CANCER_IMMORTAL].astype(float))
    return _EXAMPLES


def example_strip(ax, ex, show_prol, head, head_col):
    """One CpG: other primary types, the five skin fibroblast lines and (optionally) 30 proliferating lines."""
    row = ex["row"]
    if show_prol:
        groups = [("other", "other\ntypes", C_OTHER), ("fib", "skin\nfibro.", C_ID), ("prol", "prolif.\nlines", C_HIST)]
    else:
        groups = [("other", "20 other\nprimary\ncell types", C_OTHER), ("fib", "5 skin\nfibroblast\nlines", C_ID)]
    for i, (k, lab, col) in enumerate(groups):
        v = ex[k].to_numpy()
        jit = np.random.default_rng(i + 11).uniform(-0.17, 0.17, len(v))
        ax.scatter(i + jit, v, s=30, color=col, edgecolor="white", linewidths=0.5, zorder=3)
    ax.axhline(row.threshold, color=C_TEXT, lw=1.0, ls="--", zorder=2)
    if not show_prol:
        ax.text(0.0, row.threshold + (0.025 if row.direction == "high" else -0.025), "threshold", ha="center",
                va="bottom" if row.direction == "high" else "top", fontsize=FS["small"], color=C_TEXT)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=FS["small"])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(head, fontsize=FS["note"] + 0.5, fontweight="bold", color=head_col, loc="left", pad=6)
    ygrid(ax)
    clean(ax)


# ======================================================================
# Figure 1: the question and the instrument
# ======================================================================
def panel_frame(fig, x0, y0, x1, y1, edge, fill="#F7F8FB"):
    """A rounded frame behind a schematic panel, in figure coordinates."""
    fig.patches.append(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.015",
                                      transform=fig.transFigure, facecolor=fill, edgecolor=edge, lw=2.2, zorder=-5))


def lollipop(ax, x, y, methylated, color, size=170, base=None):
    base = y - 0.13 if base is None else base
    ax.plot([x, x], [base, y], color=color, lw=2.4, zorder=5, solid_capstyle="butt")
    ax.scatter([x], [y], s=size, facecolor=color if methylated else "white", edgecolor=color, linewidths=2.6, zorder=6)


def bound(ax, x, curve, xc, hw, h, face, edge, label, tcol, lobes=(), p=2.2, lift=0.012, label_dy=None):
    """A protein sitting on a curve: a dome whose base follows the curve, plus round lobes (dx, dy, r) above the
    base centre. Sizes are in inches so shapes stay round on unequal axes; each part is drawn twice, outlines first,
    so overlapping parts share one outline. Call after the axis limits are set."""
    fig, bb = ax.figure, ax.get_position()
    ux = np.ptp(ax.get_xlim()) / (bb.width * fig.get_figwidth())          # data units per inch
    uy = np.ptp(ax.get_ylim()) / (bb.height * fig.get_figheight())
    t = np.linspace(-1, 1, 101)
    xs = xc + t * hw * ux
    floor = np.interp(xs, x, curve) + lift * uy
    top = floor + h * uy * (1 - np.abs(t) ** p) ** (1 / p)
    y0 = np.interp(xc, x, curve) + lift * uy
    parts = lambda **kw: ([Polygon(np.column_stack([np.r_[xs, xs[::-1]], np.r_[floor, top[::-1]]]), **kw)]
                          + [Ellipse((xc + dx * ux, y0 + dy * uy), 2 * r * ux, 2 * r * uy, **kw) for dx, dy, r in lobes])
    for pt in parts(facecolor=face, edgecolor=edge, lw=3.2, zorder=4.5) + parts(facecolor=face, lw=0, zorder=4.6):
        ax.add_patch(pt)
    dy = h / 2 if label_dy is None else label_dy
    ax.text(xc, y0 + dy * uy, label, ha="center", va="center", fontsize=FS["small"], fontweight="bold", color=tcol,
            zorder=4.7)


def pill(ax, x, y, text, sub, edge, w=0.34, h=None, fs=None, ls="-"):
    lines = text.count("\n") + 1
    h = h or 0.075 + 0.045 * lines
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="white", edgecolor=edge, lw=2.0, ls=ls, zorder=3))
    ax.text(x, y + 0.02, text, ha="center", va="center", fontsize=fs or FS["small"] + 0.5, color=C_TEXT, zorder=4,
            linespacing=1.15)
    ax.text(x, y - h / 2 + 0.022, sub, ha="center", va="center", fontsize=FS["small"] - 1.5, color=C_MUTED, zorder=4)


def fig_concept():
    fig = plt.figure(figsize=(14, 10.5))
    C_FIB, C_OTH_LINE = "#1F3F75", "#8C939E"
    panel_frame(fig, 0.02, 0.52, 0.635, 0.985, C_FIB)
    panel_frame(fig, 0.655, 0.52, 0.98, 0.985, C_MUTED)
    panel_frame(fig, 0.02, 0.015, 0.98, 0.495, C_TEXT)

    # ---- A: fibroblast-unique CpGs and the two directions of movement
    ax = fig.add_axes([0.06, 0.565, 0.56, 0.33])
    x = np.linspace(0, 10.8, 650)
    dip = lambda c, s: np.exp(-0.5 * ((x - c) / s) ** 6)       # flat-bottomed, so a protein can sit on the floor
    base = 0.84 + 0.03 * np.sin(1.3 * x)
    X_UL, X_UH = 2.9, 7.4
    others = base + (0.14 - base) * dip(X_UH, 0.95)
    fib = base + 0.02 + (0.065 - base - 0.02) * dip(2.62, 1.05)
    ax.fill_between(x, others - 0.07, np.minimum(others + 0.07, 1), color=C_OTHER, alpha=0.45, lw=0, zorder=1)
    ax.plot(x, others, ls=(0, (5, 3)), color=C_OTH_LINE, lw=3, zorder=2)
    ax.plot(x, fib, color=C_FIB, lw=3.6, zorder=3)
    i_ul, i_uh = np.argmin(np.abs(x - X_UL)), np.argmin(np.abs(x - X_UH))
    lollipop(ax, X_UL, fib[i_ul], False, C_FIB)
    lollipop(ax, X_UH, fib[i_uh], True, C_FIB)
    for xc, y0, y1, (xt, yt) in ((X_UL + 0.25, fib[i_ul] + 0.07, others[i_ul] - 0.06, (4.15, 0.42)),
                                 (X_UH + 0.7, fib[i_uh] - 0.06, others[i_uh] + 0.24, (8.3, 0.12))):
        ax.annotate("", xy=(xc, y1), xytext=(xc, y0), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                                                                      color=C_TEXT, lw=2.4))
        ax.text(xt, yt, "toward other cell types\n= identity loss  (+)", ha="left", va="center", fontsize=FS["small"],
                color=C_TEXT, fontweight="bold")
    ax.annotate("", xy=(X_UL + 0.25, -0.075), xytext=(X_UL + 0.25, fib[i_ul] - 0.035),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.4", color=C_MUTED, lw=1.8))
    ax.text(X_UL + 0.4, -0.02, "away (−)", ha="left", va="center", fontsize=FS["small"] - 0.5, color=C_MUTED)
    ax.annotate("", xy=(6.9, 1.02), xytext=(6.9, fib[i_uh] + 0.0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.4", color=C_MUTED, lw=1.8))
    ax.text(6.75, 0.99, "away (−)", ha="right", va="center", fontsize=FS["small"] - 0.5, color=C_MUTED)
    ax.text(2.62, -0.16, "uniquely low (UL)", ha="center", va="top", fontsize=FS["note"], fontweight="bold",
            color=C_FIB)
    ax.text(X_UH, -0.16, "uniquely high (UH)", ha="center", va="top", fontsize=FS["note"], fontweight="bold",
            color=C_FIB)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(-0.08, 1.06)
    bound(ax, x, fib, 2.27, 0.2, 0.3, "#C9D7EE", C_FIB, "TF", C_FIB)               # keeps the fibroblast site open
    bound(ax, x, others, X_UH, 0.29, 0.17, "#DFE2E7", "#6E7682", "PRC2", "#454C58", p=2.6,
          lobes=((-0.13, 0.22, 0.155), (0.13, 0.22, 0.155)), label_dy=0.17)             # marks the site in other types
    ax.set_yticks([0, 1])
    ax.set_ylabel("Methylation (β)", fontsize=FS["lab"])
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=C_FIB, lw=3.6, label="fibroblast"),
                       Line2D([], [], color=C_OTH_LINE, lw=3, ls=(0, (5, 3)), label="20 other primary cell types")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=FS["note"], frameon=False,
              handlelength=3)
    fig.text(0.035, 0.955, "A", fontsize=FS["letter"], fontweight="bold", va="center")
    fig.text(0.065, 0.955, "Fibroblast-unique CpGs, and what identity loss would look like", fontsize=FS["title"],
             fontweight="bold", color=C_TEXT, va="center")

    # ---- B: the matched background
    ax = fig.add_axes([0.70, 0.565, 0.25, 0.33])
    for xc, y1, col, lab in ((1.0, 0.42, C_FIB, "panel\nCpG"), (2.4, 0.27, C_BG, "background\nCpG")):
        ax.plot([xc, xc], [-0.06, 0.12], color=col, lw=2.4, ls=":", zorder=2)
        ax.scatter([xc], [0.12], s=150, facecolor="white", edgecolor=col, linewidths=2.2, zorder=4)
        lollipop(ax, xc + 0.28, y1, True, col, size=150, base=0.12)
        ax.annotate("", xy=(xc + 0.28, y1 - 0.035), xytext=(xc + 0.05, 0.14),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, connectionstyle="arc3,rad=-0.25"))
        ax.text(xc + 0.14, -0.12, lab, ha="center", va="top", fontsize=FS["small"], color=C_TEXT)
    ax.plot([0.75, 3.1], [0.12, 0.12], color=C_MUTED, lw=1, ls="--", zorder=1)
    ax.text(3.15, 0.12, "same\nstart", ha="left", va="center", fontsize=FS["small"] - 1, color=C_MUTED)
    ax.plot([1.55, 1.75, 1.75, 1.55], [0.42, 0.42, 0.27, 0.27], color=C_TEXT, lw=1.4)
    ax.plot([1.75, 2.65], [0.27, 0.27], color=C_BG, lw=1, ls="--")
    ax.text(1.85, 0.345, "reported\nshift", ha="left", va="center", fontsize=FS["small"], color=C_TEXT,
            fontweight="bold")
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(-0.24, 0.52)
    ax.axis("off")
    fig.text(0.665, 0.955, "B", fontsize=FS["letter"], fontweight="bold", va="center")
    fig.text(0.692, 0.955, "Beyond matched background", fontsize=FS["title"], fontweight="bold",
             color=C_TEXT, va="center")

    # ---- C: the designs, by what each adds
    ax = fig.add_axes([0.035, 0.03, 0.93, 0.40])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0, xm, x1, y0, ym, y1 = 0.17, 0.585, 1.0, 0.04, 0.46, 0.88
    for (a, b, c, d) in ((x0, xm, ym, y1), (xm, x1, ym, y1), (x0, xm, y0, ym), (xm, x1, y0, ym)):
        ax.add_patch(FancyBboxPatch((a + 0.006, c + 0.012), b - a - 0.012, d - c - 0.024,
                                    boxstyle="round,pad=0,rounding_size=0.02", facecolor="white", edgecolor=C_GRID,
                                    lw=1.5, zorder=1))
    ax.add_patch(plt.Rectangle((x0, ym), x1 - x0, y1 - ym, facecolor="#DCE5F3", lw=0, zorder=0, alpha=0.7))
    ax.add_patch(plt.Rectangle((xm, y0), x1 - xm, y1 - y0, facecolor="#F6E3C8", lw=0, zorder=0, alpha=0.7))
    ax.text((x0 + xm) / 2, y1 + 0.035, "no added DNA damage", ha="center", fontsize=FS["note"] + 0.5,
            fontweight="bold", color=C_TEXT)
    ax.text((xm + x1) / 2, y1 + 0.035, "added DNA damage (X-rays, doxorubicin, progerin)", ha="center",
            fontsize=FS["note"] + 0.5, fontweight="bold", color=C_TEXT)
    ax.text(x0 - 0.015, (ym + y1) / 2 + 0.1, "many\ndivisions", ha="right", va="center", fontsize=FS["note"] + 0.5,
            fontweight="bold", color=C_TEXT)
    ax.text(x0 - 0.015, (y0 + ym) / 2, "few\ndivisions", ha="right", va="center", fontsize=FS["note"] + 0.5,
            fontweight="bold", color=C_TEXT)
    cl, cr = (x0 + xm) / 2, (xm + x1) / 2
    pill(ax, cl, 0.76, "replicative exhaustion", "GSE197723", ARM["Replication"][1])
    pill(ax, cl, 0.58, "replicative senescence, 2 more donors", "GSE227160", C_MUTED)
    pill(ax, cl, 0.34, "RAS oncogene → senescence", "GSE197723", ARM["RAS"][1])
    pill(ax, cl, 0.16, "untreated controls", "GSE197723", "#B8BEC8", ls="--")
    pill(ax, cr - 0.095, 0.755, "chronic low-dose\nradiation, passaged", "GSE197723", C_MUTED, w=0.2)
    pill(ax, cr - 0.095, 0.575, "irradiated clones, ~20\ndoublings after 2 Gy", "GSE112812", C_MUTED, w=0.2)
    pill(ax, cr - 0.095, 0.345, "acute X-ray\n→ senescence", "GSE197723", ARM["X-ray"][1], w=0.2)
    pill(ax, cr - 0.095, 0.155, "irradiation- or doxorubicin-\ninduced senescence", "GSE227160", C_MUTED, w=0.2)
    ax.add_patch(FancyBboxPatch((cr + 0.045, 0.14), 0.15, 0.62, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="white", edgecolor=C_MUTED, lw=2.0, ls="--", zorder=3))
    ax.text(cr + 0.12, 0.47, "progeria\n(HGPS) vs\ncontrol\n\ndivisions\nunrecorded", ha="center", va="center",
            fontsize=FS["small"] + 0.5, color=C_TEXT, zorder=4)
    ax.text(cr + 0.12, 0.18, "GSE149960", ha="center", va="center", fontsize=FS["small"] - 1.5, color=C_MUTED,
            zorder=4)
    ax.text(x0 - 0.015, ym + 0.02, "If division erodes\nidentity: loss in\nthe top row", ha="right", va="bottom",
            fontsize=FS["small"], color="#2C5AA0", fontweight="bold")
    ax.text((xm + x1) / 2, y1 + 0.095, "If DNA damage erodes identity: loss in the right column", ha="center",
            va="bottom",
            fontsize=FS["small"], color="#9A5A18", fontweight="bold")
    fig.text(0.035, 0.47, "C", fontsize=FS["letter"], fontweight="bold", va="center")
    fig.text(0.065, 0.47, "The test: experiments that separate DNA damage from cell division", fontsize=FS["title"],
             fontweight="bold", color=C_TEXT, va="center")
    save(fig, "figure_1_concept")


# ======================================================================
# Figure 2: the whole panel, then its two layers
# ======================================================================
BIN_ORDER = ["0", "0-0.1", "0.1-0.2", "0.2-0.4", "0.4-0.7", "0.7-1"]
BIN_LABELS = ["0", "0–0.1", "0.1–0.2", "0.2–0.4", "0.4–0.7", "0.7–1"]


def quad_strips(ax, panels, offsets, ylim=None, fs_mean=None):
    """Per-pair shifts toward other cell types for the three GSE197723 arms; panels = (panel, d, colour, marker).

    Above each group, outside the axes: mean shift, pairs moving toward other cell types / 24, Wilcoxon p.
    """
    w = min(0.26, abs(offsets[1] - offsets[0]) * 0.8)
    for i, arm in enumerate(ARM_ORDER):
        for k, ((panel, d, col, mk), off) in enumerate(zip(panels, offsets)):
            v = -per_rows("GSE197723", f"{arm} vs control", d, panel).excess_diff.to_numpy()
            strip(ax, i + off, v, col, marker=mk, s=24, w=w, seed=i * 10 + k)
            e = ex_row("GSE197723", f"{arm} vs control", d, panel)
            sig = e.p < 0.05
            n_toward = int(round(float(e.n))) - int(e.n_positive)
            kw = dict(xycoords=("data", "axes fraction"), textcoords="offset points", ha="center", va="bottom",
                      color=col if sig else C_MUTED, annotation_clip=False)
            ax.annotate("{:+.3f}".format(-e.excess_shift), xy=(i + off, 1.0), xytext=(0, 25),
                        fontsize=fs_mean or FS["res"] - 1.5, fontweight="bold" if sig else "normal", **kw)
            ax.annotate("{}/24\n{}".format(n_toward, pfmt(e.p)), xy=(i + off, 1.0), xytext=(0, 2),
                        fontsize=FS["small"] - 1.5, linespacing=1.05, **kw)
    ax.axhline(0, color=C_TEXT, lw=1.0, zorder=2)
    ax.set_xticks(range(len(ARM_ORDER)))
    ax.set_xticklabels([ARM[a][0] for a in ARM_ORDER], fontsize=FS["lab"])
    ax.set_xlim(-0.55, len(ARM_ORDER) - 0.45)
    if ylim:
        ax.set_ylim(*ylim)
    ygrid(ax)
    clean(ax)


def fig_two_layers():
    fig = plt.figure(figsize=(14, 16.5))
    g1 = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.55, 0.55], left=0.085, right=0.985, top=0.915, bottom=0.735,
                          wspace=0.3)
    g2 = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.0], left=0.085, right=0.955, top=0.615, bottom=0.445,
                          wspace=0.42)
    g3 = fig.add_gridspec(1, 1, left=0.085, right=0.985, top=0.30, bottom=0.05)
    n_all = {d: int(SUMM[(SUMM.panel == "fibroblast_skin") & (SUMM.direction == d)].n.sum()) for d in ("low", "high")}

    # ---- A: the whole panel
    ax = fig.add_subplot(g1[0, 0])
    quad_strips(ax, [("fibroblast:all", "low", C_UL, "o"), ("fibroblast:all", "high", C_UH, "D")], (-0.2, 0.2),
                ylim=(-0.215, 0.03))
    ax.set_ylabel(TOWARD, fontsize=FS["lab"])
    ax.legend(handles=[Line2D([], [], marker="o", color=C_UL, lw=0, ms=8, label=f"UL ({n_all['low']:,} CpGs)"),
                       Line2D([], [], marker="D", color=C_UH, lw=0, ms=7, label=f"UH ({n_all['high']:,} CpGs)")],
              loc="lower left", fontsize=FS["note"], frameon=False)
    title(ax, "The whole panel (9,150 CpGs): exhausted cells\nmove away from other cell types", pad=46)
    letter(ax, "A", -0.14, 1.32)

    # ---- B: one identity and one history CpG
    ex = example_cpgs()
    for j, (k, head, col) in enumerate((("identity UL", "identity CpG", C_ID),
                                         ("history UL", "history CpG", C_HIST))):
        ax = fig.add_subplot(g1[0, 1 + j])
        share = int(round(ex[k]["row"].cancer_side_frac * 30))
        example_strip(ax, ex[k], True, f"{head}\n{share}/30 prolif. lines\nshare its state", col)
        if j == 0:
            ax.set_ylabel("Methylation (β)", fontsize=FS["lab"])
            letter(ax, "B", -0.45, 1.36)
            ax.text(-0.12, 1.3, "Two kinds of unique CpG", transform=ax.transAxes, fontsize=FS["title"],
                    fontweight="bold", color=C_TEXT, va="bottom")
        else:
            ax.set_yticklabels([])

    # ---- C, D: the gradient
    for j, (col_, ttl, ylab, sign) in enumerate([
            ("excess_shift", "Movement beyond background", TOWARD, -1.0),
            ("raw_signed_shift", "Raw methylation change", "Δβ, treated − control\n(+ = gain of methylation)", None)]):
        ax = fig.add_subplot(g2[0, j])
        ax.axvspan(-0.5, 0.45, color="#E3EAF5", zorder=0)
        ax.axvspan(2.55, 5.5, color="#FBEFD9", zorder=0)
        ax.axhline(0, color="#444444", lw=0.9, zorder=1)
        x = np.arange(len(BIN_ORDER))
        for arm in ARM_ORDER:
            lab, c, mk = ARM[arm]
            for d, ls, fill in (("low", "-", True), ("high", "--", False)):
                g = BINS[(BINS.arm == arm) & (BINS.direction == d)].set_index("bin").reindex(BIN_ORDER)
                y = g[col_].to_numpy() * (sign if sign else (-1.0 if d == "low" else 1.0))
                ax.plot(x, y, ls, color=c, lw=1.9, zorder=3, alpha=0.95 if fill else 0.75)
                ax.scatter(x, y, s=44 if fill else 38, marker=mk, zorder=4, facecolor=c if fill else "white",
                           edgecolor=c, linewidth=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, fontsize=FS["small"])
        ax.set_xlim(-0.5, 5.5)
        ax.set_xlabel("Share of 30 proliferating lines\nthat also hold the fibroblast state", fontsize=FS["lab"] - 1)
        ax.set_ylabel(ylab, fontsize=FS["lab"] - 0.5)
        ygrid(ax)
        clean(ax)
        top = ax.get_ylim()[1]
        ax.text(0.0, top, "identity", ha="center", va="bottom", fontsize=FS["note"], color=C_ID, fontweight="bold")
        ax.text(4.0, top, "proliferation history", ha="center", va="bottom", fontsize=FS["note"], color=C_HIST,
                fontweight="bold")
        title(ax, ttl, pad=24)
        letter(ax, "CD"[j], -0.24, 1.2)
    handles = [Line2D([], [], color=ARM[a][1], marker=ARM[a][2], lw=1.9, ms=7, label=ARM[a][0]) for a in ARM_ORDER]
    handles += [Line2D([], [], color="#444444", lw=1.9, label="UL"),
                Line2D([], [], color="#444444", lw=1.9, ls="--", label="UH")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.085, 0.652), ncol=5, frameon=False,
               fontsize=FS["note"], handlelength=2.2, columnspacing=1.6)

    # ---- E: reproducibility in an independent batch, against chance
    ax = fig.add_subplot(g2[0, 2])
    rows = [("identity", "low", "Identity UL"), ("history", "low", "History UL"),
            ("identity", "high", "Identity UH"), ("history", "high", "History UH")]
    ys = np.array([3.5, 2.5, 1.0, 0.0])
    summ = SUMM[SUMM.panel == "fibroblast_skin"].set_index(["layer", "direction"])
    for yy, (lay, d, lab) in zip(ys, rows):
        g = summ.loc[(lay, d)]
        c = C_ID if lay == "identity" else C_HIST
        ax.barh(yy, g.replicates_gse197723, height=0.62, color=c, alpha=0.9 if d == "low" else 0.5, zorder=3)
        xt = max(g.replicates_gse197723, g.replicates_chance) + 0.03
        ax.text(xt, yy + 0.12, "{:.0%}".format(g.replicates_gse197723), va="center", fontsize=FS["note"],
                fontweight="bold", color=c)
        ax.text(xt, yy - 0.2, ptext(g.replicates_p_vs_chance), va="center", fontsize=FS["small"] - 1.5, color=C_TEXT)
        ax.text(-0.04, yy, "{}\n{:,} CpGs".format(lab, int(g.n)), va="center", ha="right", fontsize=FS["small"],
                color=C_TEXT, linespacing=1.2)
    for d, (ya, yb) in (("low", (3.5, 2.5)), ("high", (1.0, 0.0))):
        ch = summ.loc[("identity", d)].replicates_chance
        ax.plot([ch, ch], [yb - 0.45, ya + 0.45], color=C_TEXT, lw=1.4, ls=(0, (4, 2.5)), zorder=5)
        ax.text(ch, ya + 0.5, "chance {:.1%}".format(ch), ha="left", va="bottom", fontsize=FS["small"] - 1,
                color=C_TEXT)
        pf = summ.loc[("identity", d)].replicates_p_identity_vs_history
        xb = 1.42
        ax.plot([xb - 0.03, xb, xb, xb - 0.03], [ya, ya, yb, yb], color=C_TEXT, lw=1.1, clip_on=False)
        ax.text(xb + 0.03, (ya + yb) / 2, pfmt(pf), ha="left", va="center", fontsize=FS["small"] - 1.5,
                color=C_TEXT, rotation=90, clip_on=False)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 4.4)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Share of CpGs whose fibroblast-unique\ncall reproduces in an independent batch",
                  fontsize=FS["lab"] - 1)
    clean(ax, left=False)
    xgrid(ax)
    ax.legend(handles=[Patch(color=C_MUTED, label="GSE197723 fibroblasts meet the rule again"),
                       Line2D([], [], color=C_TEXT, lw=1.4, ls=(0, (4, 2.5)), label="expected by chance")],
              loc="upper center", bbox_to_anchor=(0.4, -0.3), fontsize=FS["small"] - 1, frameon=False)
    title(ax, "Reproducibility", pad=24)
    letter(ax, "E", -0.45, 1.2)

    # ---- F: the same pairs, split
    ax = fig.add_subplot(g3[0, 0])
    panels = [("fibroblast:identity", "low", C_ID, "o"), ("fibroblast:identity", "high", C_ID, "D"),
              ("fibroblast:history", "low", C_HIST, "o"), ("fibroblast:history", "high", C_HIST, "D")]
    quad_strips(ax, panels, (-0.33, -0.11, 0.11, 0.33), ylim=(-0.36, 0.15))
    ax.set_ylabel(TOWARD, fontsize=FS["lab"])
    nn = {(lay, d): int(ex_row("GSE197723", "X-ray vs control", d, f"fibroblast:{lay}").n_probes)
          for lay in ("identity", "history") for d in ("low", "high")}
    ax.legend(handles=[Line2D([], [], marker="o", color=C_ID, lw=0, ms=8,
                              label=f"identity UL ({nn[('identity', 'low')]:,} CpGs)"),
                       Line2D([], [], marker="D", color=C_ID, lw=0, ms=7,
                              label=f"identity UH ({nn[('identity', 'high')]:,})"),
                       Line2D([], [], marker="o", color=C_HIST, lw=0, ms=8,
                              label=f"history UL ({nn[('history', 'low')]:,})"),
                       Line2D([], [], marker="D", color=C_HIST, lw=0, ms=7,
                              label=f"history UH ({nn[('history', 'high')]:,})")],
              loc="lower left", bbox_to_anchor=(0.0, 0.06), ncol=4, fontsize=FS["note"], frameon=False,
              columnspacing=1.4)
    title(ax, "The same 24 pairs, split: identity CpGs move toward other cell types, history CpGs away from them",
          pad=46)
    letter(ax, "F", -0.075, 1.21)
    save(fig, "figure_2_two_layers")


# ======================================================================
# Figure 3: identity information lost
# ======================================================================
def fig_information():
    fig = plt.figure(figsize=(14, 11.5))
    g1 = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15], left=0.06, right=0.985, top=0.93, bottom=0.57,
                          wspace=0.22)
    g2 = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.13, right=0.985, top=0.44, bottom=0.08, wspace=0.55)

    # ---- A: the measure, CpG by CpG
    ax = fig.add_subplot(g1[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    letter(ax, "A", 0.0, 1.03)
    ax.text(0.07, 1.0, "How far each CpG moved, weighted", fontsize=FS["title"], fontweight="bold", color=C_TEXT,
            va="top")
    xl, xr = 0.25, 0.80
    ax.text(xl, 0.84, "fibroblast", ha="center", va="bottom", fontsize=FS["small"], color=C_ID, fontweight="bold")
    ax.text(xr, 0.84, "other cell\ntypes", ha="center", va="bottom", fontsize=FS["small"], color=C_MUTED,
            fontweight="bold")
    for y, lab, lw, f in ((0.74, "UL CpG", 9.0, 0.55), (0.585, "UH CpG", 4.5, 0.25), (0.43, "UL CpG", 2.0, -0.15)):
        ax.plot([xl, xr], [y, y], color="#DADDE3", lw=lw, solid_capstyle="round", zorder=1)
        xf = xl + f * (xr - xl)
        ax.plot([xl, xf], [y, y], color=C_TEXT if f > 0 else C_MUTED, lw=lw, solid_capstyle="butt", zorder=2)
        ax.scatter([xl], [y], s=80, color=C_ID, zorder=3, edgecolor="white")
        ax.scatter([xr], [y], s=80, color=C_OTHER, edgecolor=C_MUTED, zorder=3)
        ax.annotate("", xy=(xf, y + 0.012), xytext=(xf, y + 0.065),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5", color=C_TEXT, lw=1.6))
        ax.text(0.05, y, lab, ha="left", va="center", fontsize=FS["small"], color=C_TEXT)
        ax.text(xr + 0.05, y, "f = {:+.2f}".format(f), ha="left", va="center", fontsize=FS["small"], color=C_TEXT)
    ax.plot([xl, xr], [0.32, 0.32], color=C_TEXT, lw=1)
    for v in (0, 0.5, 1):
        ax.plot([xl + v * (xr - xl)] * 2, [0.31, 0.33], color=C_TEXT, lw=1)
        ax.text(xl + v * (xr - xl), 0.29, "{:g}".format(v), ha="center", va="top", fontsize=FS["small"], color=C_TEXT)
    ax.text((xl + xr) / 2, 0.22, "f: fraction of the way moved", ha="center", va="top", fontsize=FS["small"],
            color=C_TEXT)
    ax.add_patch(FancyBboxPatch((0.05, 0.015), 0.9, 0.11, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="#F4F5F7", edgecolor=C_GRID, lw=1))
    ax.text(0.5, 0.07, "information lost = Σ wᵢ fᵢ / Σ wᵢ  − background", ha="center", va="center",
            fontsize=FS["note"] + 0.5, fontweight="bold", color=C_TEXT)
    ax.text(0.05, 0.345, "line width = weight wᵢ", ha="left", va="center", fontsize=FS["small"] - 1, color=C_MUTED,
            style="italic")

    # ---- B: the 24 cultures
    ax = fig.add_subplot(g1[0, 1])
    xt = []
    for a, arm in enumerate(ARM_ORDER):
        for k, lay in enumerate(("identity", "history")):
            c, mk = LAYER_MK[lay]
            x0 = a * 2.8 + k * 1.15
            v = pls_rows("GSE197723", f"{arm} vs control", lay).frac_lost.to_numpy()
            strip(ax, x0, v, c, marker=mk, s=26, w=0.66, seed=a * 3 + k)
            r = ils_row("GSE197723", f"{arm} vs control", lay)
            kw = dict(xycoords=("data", "axes fraction"), textcoords="offset points", ha="center", va="bottom",
                      color=c, annotation_clip=False)
            ax.annotate("{:+.3f}".format(r.frac_lost), xy=(x0, 1.0), xytext=(0, 14), fontsize=FS["res"],
                        fontweight="bold", **kw)
            ax.annotate(pfmt(r.p), xy=(x0, 1.0), xytext=(0, 2), fontsize=FS["small"] - 1.5, **kw)
            xt.append(x0)
        ax.text(a * 2.8 + 0.575, -0.1, ARM[arm][0], ha="center", va="top", fontsize=FS["lab"], color=C_TEXT,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.set_xticks(xt)
    ax.set_xticklabels(["identity", "history"] * 3, fontsize=FS["small"])
    ax.set_xlim(-0.6, 2 * 2.8 + 1.75)
    ax.set_ylim(-0.75, 0.6)
    ax.set_ylabel(LOST, fontsize=FS["lab"])
    ygrid(ax)
    clean(ax)
    title(ax, "The same 24 matched pairs, as information", pad=34)
    letter(ax, "B", -0.13, 1.16)

    # ---- C: the bound
    ax = fig.add_subplot(g2[0, 0])
    items = ["Replication", "RAS", "X-ray"]
    xray = ils_row("GSE197723", "X-ray vs control", "identity")
    ys = np.arange(len(items))[::-1]
    for yy, arm in zip(ys, items):
        r = ils_row("GSE197723", f"{arm} vs control", "identity")
        ax.barh(yy, r.frac_lost, height=0.5, color=C_ID, alpha=0.9, zorder=3)
        ax.errorbar(r.frac_lost, yy, xerr=[[r.frac_lost - r.frac_lost_lo], [r.frac_lost_hi - r.frac_lost]],
                    fmt="none", ecolor=C_TEXT, elinewidth=1.2, capsize=3, zorder=4)
        x_txt = max(r.frac_lost_hi, 0) + 0.01
        ax.text(x_txt, yy + 0.1, "{:+.3f}".format(r.frac_lost), va="center", fontsize=FS["res"], fontweight="bold",
                color=C_ID)
        ax.text(x_txt, yy - 0.18, "95% CI {:+.3f} to {:+.3f}".format(r.frac_lost_lo, r.frac_lost_hi), va="center",
                fontsize=FS["small"], color=C_MUTED)
    ax.axvspan(0, xray.mde80, ymin=0, ymax=0.28, color="#F2D9A6", alpha=0.7, zorder=1)
    ax.text(xray.mde80 + 0.004, -0.42, "smallest loss these 24 pairs would detect\n(80% power): {:.3f}".format(
        xray.mde80), fontsize=FS["small"], color="#9A6B12", va="top")
    ax.axvline(0, color=C_TEXT, lw=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([ARM[a][0] for a in items], fontsize=FS["lab"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.05, 0.36)
    ax.set_ylim(-0.75, len(items) - 0.4)
    ax.set_xlabel("Identity information lost (fraction)\n" + ARROWS, fontsize=FS["lab"])
    clean(ax, left=False)
    xgrid(ax)
    title(ax, "The X-ray result is a bounded negative", pad=10)
    letter(ax, "C", -0.42, 1.1)

    # ---- D: what the background subtraction does, X-ray
    ax = fig.add_subplot(g2[0, 1])
    comb = ils_row("GSE197723", "X-ray vs control", "identity")
    ul, uh = (ex_row("GSE197723", "X-ray vs control", d, "fibroblast:identity") for d in ("low", "high"))
    groups = [("identity layer,\nUL + UH combined", comb.frac_lost_raw, comb.frac_lost_bg, comb.frac_lost,
               comb.p_raw, comb.p),
              (f"UL half\n({int(ul.loss_n_probes)} CpGs)", ul.loss_frac_raw, ul.loss_frac_bg, ul.loss_frac,
               ul.loss_p_raw, ul.loss_p),
              (f"UH half\n({int(uh.loss_n_probes)} CpGs)", uh.loss_frac_raw, uh.loss_frac_bg, uh.loss_frac,
               uh.loss_p_raw, uh.loss_p)]
    ys = np.arange(len(groups))[::-1] * 1.25
    for yy, (lab, raw, bg, net, p_raw, p_net) in zip(ys, groups):
        for off, val, kw, txt, p_ in ((0.32, raw, dict(facecolor="white", edgecolor=C_ID, hatch="////"),
                                       "panel CpGs' own shift", p_raw),
                                      (0.0, bg, dict(facecolor=C_BG, edgecolor=C_BG), "background term", None),
                                      (-0.32, net, dict(facecolor=C_ID, edgecolor=C_ID), "net = panel − background",
                                       p_net)):
            ax.barh(yy + off, val, height=0.28, lw=1.2, zorder=3, **kw)
            s = "{:+.3f}".format(val) + (f"  p = {pfmt(p_)}" if p_ is not None else "")
            ax.text(0.0235, yy + off, s, va="center", ha="left", fontsize=FS["small"] - 0.5, color=C_TEXT)
    ax.axvline(0, color=C_TEXT, lw=1.0, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([g[0] for g in groups], fontsize=FS["note"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.024, 0.05)
    ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
    ax.set_xlabel("Identity information lost (fraction)\n" + ARROWS, fontsize=FS["lab"])
    clean(ax, left=False)
    xgrid(ax)
    ax.legend(handles=[Patch(facecolor="white", edgecolor=C_ID, hatch="////", label="panel CpGs' own shift"),
                       Patch(color=C_BG, label="background term"),
                       Patch(color=C_ID, label="net = panel − background")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=FS["small"], frameon=False,
              handletextpad=0.4, columnspacing=1.0)
    title(ax, "Acute X-ray: the apparent loss is background drift", pad=10)
    letter(ax, "D", -0.37, 1.1)
    save(fig, "figure_3_information_loss")


# ======================================================================
# Figure 4: every perturbation
# ======================================================================
LOSS_SPEC = [  # label, category, dataset, contrast, n label
    ("Irradiated clones (2 Gy, ~20 doublings)", 0, "GSE112812", C1_ALL, "35 vs 11"),
    ("   same, excl. 2 outlier controls", 0, "GSE112812", C1_EXCL, "35 vs 9"),
    ("   donor 2, not fibroblast-like", 0, "GSE112873", C2, "26 vs 20"),
    ("Chronic low-dose radiation †", 0, "GSE197723", "chronic radiation vs untreated", "4 passage pairs"),
    ("Acute X-ray → senescence", 1, "GSE197723", "X-ray vs control", "24 pairs"),
    ("IR-induced senescence (DSF2)", 1, "GSE227160", "DSF2: IRR_SIPS vs young", "2 vs 1"),
    ("IR-induced senescence (DSF3)", 1, "GSE227160", "DSF3: IRR_SIPS vs young", "2 vs 1"),
    ("Doxorubicin senescence (DSF2)", 1, "GSE227160", "DSF2: Dox_SIPS vs young", "2 vs 1"),
    ("Doxorubicin senescence (DSF3)", 1, "GSE227160", "DSF3: Dox_SIPS vs young", "2 vs 1"),
    ("RAS oncogene-induced senescence", 2, "GSE197723", "RAS vs control", "24 pairs"),
    ("Replicative exhaustion", 2, "GSE197723", "Replication vs control", "24 pairs"),
    ("Replicative senescence (DSF2)", 2, "GSE227160", "DSF2: RS vs young", "2 vs 1"),
    ("Replicative senescence (DSF3)", 2, "GSE227160", "DSF3: RS vs young", "2 vs 1"),
    ("Progeria (HGPS) vs control", 2, "GSE149960", "HGPS vs control", "9 vs 6 people"),
]
CAT_NAMES = ["DAMAGE, THEN CONTINUED DIVISION", "DAMAGE, THEN GROWTH ARREST",
             "REPLICATIVE OR ONCOGENIC STRESS, PROGERIA"]
CAT_FILL = ["#FBEFD9", "#EEF0F4", "#E3F1E8"]
CAT_TEXT = ["#9A5A18", "#5A6478", "#2E7D55"]


def forest(ax, spec, xlo, xhi, show_bg=True, pcols=True, fs_row=None):
    fs_row = fs_row or FS["note"] + 0.5
    ypos, y, prev = {}, 0.0, None
    for i, (_, cat, *_r) in enumerate(spec):
        if prev is not None and cat != prev:
            y -= 0.9
        ypos[i], prev, y = y, cat, y - 1
    w = xhi - xlo
    span = {}
    for i, (_, cat, *_r) in enumerate(spec):
        span.setdefault(cat, []).append(ypos[i])
    for cat, yy in span.items():
        ax.axhspan(min(yy) - 0.5, max(yy) + 0.52, color=CAT_FILL[cat], zorder=0)
        ax.text(xlo - 0.025 * w, max(yy) + 0.58, CAT_NAMES[cat], fontsize=FS["small"], fontweight="bold",
                color=CAT_TEXT[cat], va="bottom", ha="right")
    ax.axvline(0, color="#333333", lw=1.1, zorder=2)
    for i, (lab, cat, ds, con, nlab) in enumerate(spec):
        yy = ypos[i]
        ax.text(xlo - 0.025 * w, yy, lab.strip(), fontsize=fs_row, color=C_TEXT, va="center", ha="right",
                style="italic" if lab.startswith(" ") else "normal")
        for k, (lay, off) in enumerate((("identity", 0.2), ("history", -0.2))):
            c, mk = LAYER_MK[lay]
            r = ils_row(ds, con, lay)
            if r is None:
                continue
            sig = pd.notna(r.p) and r.p < 0.05
            ax.plot([0, r.frac_lost], [yy + off] * 2, color=c, lw=1.7, alpha=0.7, zorder=3)
            if show_bg:
                ax.scatter([r.frac_lost_bg], [yy + off], s=36, marker="x", color=C_BG, zorder=4, linewidth=1.3)
            ax.scatter([r.frac_lost], [yy + off], s=75, marker=mk, zorder=5, facecolor=c if sig else "white",
                       edgecolor=c, linewidth=1.7)
            if pcols:
                ax.text(xhi + (0.52 + 0.2 * k) * w, yy, pfmt(r.p) if pd.notna(r.p) else "—", fontsize=FS["small"],
                        ha="center", va="center", color=c, fontweight="bold" if sig else "normal")
        if pcols:
            ax.text(xhi + 0.03 * w, yy, ds, fontsize=FS["small"] - 1.5, color=C_MUTED, va="center", ha="left")
            ax.text(xhi + 0.2 * w, yy, nlab, fontsize=FS["small"], color=C_TEXT, va="center", ha="left")
    top = max(ypos.values())
    if pcols:
        ax.text(xhi + 0.2 * w, top + 1.0, "n", fontsize=FS["small"], fontweight="bold", color=C_TEXT)
        ax.text(xhi + 0.52 * w, top + 1.0, "p identity", fontsize=FS["small"], fontweight="bold", color=C_ID,
                ha="center")
        ax.text(xhi + 0.72 * w, top + 1.0, "p history", fontsize=FS["small"], fontweight="bold", color=C_HIST,
                ha="center")
    ax.set_ylim(min(ypos.values()) - 0.8, top + 1.5)
    ax.set_xlim(xlo, xhi)
    ax.set_yticks([])
    clean(ax, left=False)
    return ypos


def fig_all():
    fig = plt.figure(figsize=(14, 17))
    g1 = fig.add_gridspec(1, 1, left=0.30, right=0.665, top=0.925, bottom=0.53)
    g2 = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 1.2], left=0.075, right=0.985, top=0.39, bottom=0.07,
                          wspace=0.42)

    # ---- A: forest
    ax = fig.add_subplot(g1[0, 0])
    forest(ax, LOSS_SPEC, -0.62, 0.30)
    ax.set_xticks(np.arange(-0.6, 0.31, 0.2))
    ax.set_xlabel("Identity information lost (fraction)\n" + ARROWS, fontsize=FS["lab"])
    ax.legend(handles=[Line2D([], [], color=C_ID, marker="o", lw=0, ms=9,
                              label=f"identity CpGs ({n_layer('identity', 'low')} UL / {n_layer('identity', 'high')} UH)"),
                       Line2D([], [], color=C_HIST, marker="s", lw=0, ms=8,
                              label=f"proliferation-history CpGs ({n_layer('history', 'low'):,} UL / "
                                    f"{n_layer('history', 'high'):,} UH)"),
                       Line2D([], [], color=C_BG, marker="x", lw=0, ms=8, mew=1.4, label="background term (subtracted)"),
                       Line2D([], [], color="#555555", marker="o", lw=0, ms=9, mfc="white", mew=1.6,
                              label="open: p ≥ 0.05 or not testable")],
              loc="lower left", bbox_to_anchor=(-0.62, 1.01), ncol=2, fontsize=FS["note"], frameon=False,
              columnspacing=1.6)
    ax.text(-0.62, 1.14, "Every perturbation, on one scale", transform=ax.transAxes, fontsize=FS["title"] + 1,
            fontweight="bold", color=C_TEXT)
    letter(ax, "A", -0.70, 1.16)

    # ---- B: GSE227160 by trigger
    ax = fig.add_subplot(g2[0, 0])
    arms = [("Middle_passaged", "middle\npassage"), ("RS", "replicative\nsenesc."),
            ("IRR_SIPS", "IR\nsenesc."), ("Dox_SIPS", "doxorubicin\nsenesc.")]
    for i, (arm, _) in enumerate(arms):
        for k, lay in enumerate(("identity", "history")):
            c, mk = LAYER_MK[lay]
            for dn, fill, dx in (("DSF2", True, -0.07), ("DSF3", False, 0.07)):
                r = ils_row("GSE227160", f"{dn}: {arm} vs young", lay)
                ax.scatter([i + (k - 0.5) * 0.36 + dx], [r.frac_lost], s=70, marker=mk, facecolor=c if fill else "white",
                           edgecolor=c, linewidths=1.6, zorder=3)
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.axvspan(1.5, 3.5, color="#EEF0F4", zorder=0)
    ax.text(2.5, 0.21, "damage-induced", ha="center", fontsize=FS["small"], color=CAT_TEXT[1], fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=FS["small"])
    ax.set_ylim(-0.25, 0.24)
    ax.set_ylabel(LOST, fontsize=FS["lab"] - 0.5)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_MUTED, lw=0, ms=8, label="donor DSF2"),
                       Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.6, lw=0, ms=8, label="donor DSF3")],
              loc="lower right", fontsize=FS["small"], frameon=False)
    ygrid(ax)
    clean(ax)
    title(ax, "Two more donors (GSE227160)\nvs each donor's young culture", pad=10, fs=FS["title"] - 1)
    letter(ax, "B", -0.3, 1.16)

    # ---- C: chronic radiation, identity layer
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=g2[0, 1], hspace=0.38)
    adj = CHRON[CHRON.panel == "fibroblast:identity"].set_index("direction")
    for k, (d, _, mk, nm) in enumerate(DIRS):
        ax = fig.add_subplot(sub[k])
        g = SER[(SER.panel == "fibroblast:identity") & (SER.direction == d)]
        u = g[g.arm == "untreated"].sort_values("pdl")
        c = g[g.arm == "chronic radiation"].sort_values("pdl")
        base = u.excess.iloc[0]
        ax.plot(u.pdl, -(u.excess - base), "-o", color=C_MUTED, ms=7, lw=2.2, label="untreated")
        ax.plot(c.pdl, -(c.excess - base), "-" + mk, color=C_ID, ms=7, lw=2.2, label="chronic radiation")
        ax.axhline(0, color=C_TEXT, lw=0.8, alpha=0.5)
        ax.set_ylabel(f"identity {nm}\nvs PDL 3 (Δβ)", fontsize=FS["lab"] - 1.5)
        ax.text(0.97, 0.05 if k == 0 else 0.93, "same passage {:+.3f}\ndoubling-matched {:+.3f}".format(
            -adj.loc[d, "raw_shift"], -adj.loc[d, "pdl_adjusted_shift"]), transform=ax.transAxes, ha="right",
            va="bottom" if k == 0 else "top", fontsize=FS["small"] - 1, color=C_ID, fontweight="bold")
        ygrid(ax)
        clean(ax)
        if k == 0:
            title(ax, "Chronic low-dose\nradiation (GSE197723)", pad=30, fs=FS["title"] - 1)
            ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 0.98), ncol=2, fontsize=FS["small"] - 1,
                      frameon=False, handlelength=1.5, columnspacing=0.8)
            ax.set_xticklabels([])
            letter(ax, "C", -0.42, 1.48)
        else:
            ax.set_xlabel("Population doublings", fontsize=FS["lab"] - 1)

    # ---- D: progeria per person, by layer and direction
    ax = fig.add_subplot(g2[0, 2])
    groups = [("identity", "low", "identity\nUL"), ("identity", "high", "identity\nUH"),
              ("history", "low", "history\nUL"), ("history", "high", "history\nUH")]
    for j, (lay, d, lab) in enumerate(groups):
        c = C_ID if lay == "identity" else C_HIST
        mk = "o" if d == "low" else "D"
        g = per_rows("GSE149960", "HGPS vs control", d, f"fibroblast:{lay}")
        for i, grp in enumerate(("ctrl", "trt")):
            v = g[g.group == grp].loss_frac.to_numpy()
            strip(ax, j * 1.0 + (i - 0.5) * 0.4, v, c, marker=mk, fill=grp == "trt", s=40, w=0.3, seed=j * 2 + i)
        e = ex_row("GSE149960", "HGPS vs control", d, f"fibroblast:{lay}")
        sig = e.loss_p < 0.05
        ax.text(j, 0.86, "{:+.3f}\np = {}".format(e.loss_frac, pfmt(e.loss_p)), ha="center", va="top",
                fontsize=FS["small"], color=c if sig else C_MUTED, fontweight="bold" if sig else "normal")
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["{}\n({:,})".format(g[2], int(ex_row("GSE149960", "HGPS vs control", g[1],
                                                                f"fibroblast:{g[0]}").loss_n_probes)) for g in groups],
                       fontsize=FS["small"])
    ax.set_ylim(-0.6, 0.9)
    ax.set_ylabel("Information lost, per CpG set (fraction)\n" + ARROWS, fontsize=FS["lab"] - 0.5)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.4, lw=0, ms=8,
                              label="control (6 people)"),
                       Line2D([], [], marker="o", color=C_MUTED, lw=0, ms=8, label="HGPS (9 people)")],
              loc="lower left", fontsize=FS["small"], frameon=False)
    ygrid(ax)
    clean(ax)
    title(ax, "Progeria vs control fibroblasts\n(GSE149960), person by person", pad=10, fs=FS["title"] - 1)
    letter(ax, "D", -0.25, 1.16)
    save(fig, "figure_4_all_perturbations")


# ======================================================================
# Figure 5: irradiated clones that kept dividing
# ======================================================================
ORIGIN = {"AG04449": "skin", "AG09309": "skin", "AG10803": "skin", "BJ": "skin", "NHDF-neo": "skin",
          "AG04450": "lung", "IMR90": "lung", "AG09319": "gum", "HCF": "heart", "ProgFib": "HGPS"}
C_SKIN, C_OFIB = "#1B2A4A", "#AEBBD0"
CLONE_GROUPS = [("ctrl", "control\nclones", C_CTRL), ("stable", "irradiated,\nstable", C_STABLE),
                ("unstable", "irradiated,\nunstable", C_UNSTABLE)]


def donor_match(ax, donor="donor1"):
    g = IDENT[IDENT.donor == donor].sort_values("r", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(g))
    cols = [C_SKIN if ORIGIN.get(c) == "skin" else C_OFIB if c in ORIGIN else C_OTHER for c in g.cell_line]
    ax.barh(y, g.r, color=cols, height=0.78, zorder=3)
    for yy, c, col in zip(y, g.cell_line, cols):
        ax.text(0.505, yy, f"{c} · {ORIGIN[c]}" if c in ORIGIN else c, va="center", fontsize=FS["small"],
                color="white" if col == C_SKIN else C_TEXT, zorder=4)
    ax.set_yticks([])
    ax.set_xlim(0.5, 0.92)
    ax.set_ylim(-0.6, len(g) - 0.4)
    ax.set_xlabel("Correlation with ENCODE cell line\n(control clones, 20k most variable CpGs)", fontsize=FS["lab"] - 1)
    xgrid(ax)
    clean(ax, left=False)


def clone_values(contrast, layer):
    p = pls_rows("GSE112812", contrast, layer).set_index("unit")
    ctrl = p[p.group == "ctrl"].index
    stab = pd.Series({s: STAB.get(s, "NA") for s in p.index})
    return {"ctrl": p.loc[ctrl, "frac_lost"],
            "stable": p.loc[(stab == "stable") & ~p.index.isin(ctrl), "frac_lost"],
            "unstable": p.loc[(stab == "unstable") & ~p.index.isin(ctrl), "frac_lost"]}


def clone_strip(ax, vals, color):
    for i, (k, lab, col) in enumerate(CLONE_GROUPS):
        v = vals[k]
        jit = np.random.default_rng(i).uniform(-0.17, 0.17, len(v))
        out = v.index.isin(OUTLIERS)
        ax.scatter((i + jit)[~out], v.values[~out], s=50, c=col, alpha=0.9, edgecolors="white", linewidths=0.7,
                   zorder=4)
        if out.any():
            ax.scatter((i + jit)[out], v.values[out], s=70, facecolors="none", edgecolors=col, linewidths=2.0,
                       zorder=4)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=color, lw=3, zorder=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lab}\n(n={len(vals[k])})" for k, lab, _ in CLONE_GROUPS], fontsize=FS["small"])
    ax.axhline(0, color=C_TEXT, lw=1.0, zorder=2)
    ygrid(ax)
    clean(ax)


def fig_clones():
    fig = plt.figure(figsize=(14, 12.5))
    g1 = fig.add_gridspec(1, 3, width_ratios=[0.85, 1, 1], left=0.06, right=0.985, top=0.92, bottom=0.56, wspace=0.42)
    g2 = fig.add_gridspec(1, 1, left=0.33, right=0.72, top=0.42, bottom=0.08)

    ax = fig.add_subplot(g1[0, 0])
    donor_match(ax)
    title(ax, "Donor 1's clones are\nskin fibroblasts", pad=10, fs=FS["title"] - 1)
    letter(ax, "A", -0.1, 1.17)

    for j, lay in enumerate(("identity", "history")):
        ax = fig.add_subplot(g1[0, 1 + j])
        c = LAYER_MK[lay][0]
        vals = clone_values(C1_ALL, lay)
        clone_strip(ax, vals, c)
        a, e = ils_row("GSE112812", C1_ALL, lay), ils_row("GSE112812", C1_EXCL, lay)
        u = stats.mannwhitneyu(vals["unstable"], vals["stable"], alternative="two-sided")   # computed here
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.55)
        for k, (t, v, p_) in enumerate((("vs all 11 controls", a.frac_lost, a.p),
                                        ("excl. clones 1-1, 1-3", e.frac_lost, e.p),
                                        ("unstable − stable", vals["unstable"].mean() - vals["stable"].mean(),
                                         u.pvalue))):
            sig = p_ < 0.05
            yy = 0.975 - k * 0.09
            ax.text(0.03, yy, t, transform=ax.transAxes, fontsize=FS["small"], va="top", color=C_TEXT)
            ax.text(0.99, yy, "{:+.3f}  p = {}".format(v, pfmt(p_)), transform=ax.transAxes, ha="right", va="top",
                    fontsize=FS["small"] + 0.5, color=c if sig else C_MUTED, fontweight="bold" if sig else "normal")
        ax.set_ylabel(LOST if j == 0 else "", fontsize=FS["lab"] - 0.5)
        title(ax, ("Identity CpGs" if lay == "identity" else "Proliferation-history CpGs"), pad=10)
        letter(ax, "BC"[j], -0.24, 1.17)

    ax = fig.add_subplot(g2[0, 0])
    spec = [("Donor 1, skin fibroblast panel, all 11 controls", 0, "GSE112812", C1_ALL, "35 vs 11", "fibroblast"),
            ("   same, excl. control clones 1-1, 1-3", 0, "GSE112812", C1_EXCL, "35 vs 9", "fibroblast"),
            ("Donor 2 (not fibroblast-like), skin panel", 1, "GSE112873", C2, "26 vs 20", "fibroblast"),
            ("Donor 1, lung fibroblast panel, all controls", 2, "GSE112812", C1_ALL, "35 vs 11", "fibroblast_lung"),
            ("   same, excl. control clones 1-1, 1-3", 2, "GSE112812", C1_EXCL, "35 vs 9", "fibroblast_lung")]
    ypos = {}
    y = 0.0
    for i, (lab, grp, ds, con, nlab, pb) in enumerate(spec):
        if i and grp != spec[i - 1][1]:
            y -= 0.6
        ypos[i] = y
        ax.text(-0.27, y, lab.strip(), ha="right", va="center", fontsize=FS["note"], color=C_TEXT,
                style="italic" if lab.startswith(" ") else "normal")
        for k, (lay, off) in enumerate((("identity", 0.17), ("history", -0.17))):
            c, mk = LAYER_MK[lay]
            r = ils_row(ds, con, lay, panel_base=pb)
            sig = r.p < 0.05
            ax.errorbar(r.frac_lost, y + off, xerr=[[r.frac_lost - r.frac_lost_lo], [r.frac_lost_hi - r.frac_lost]],
                        fmt="none", ecolor=c, elinewidth=1.3, capsize=2.5, zorder=3)
            ax.scatter([r.frac_lost], [y + off], s=70, marker=mk, facecolor=c if sig else "white", edgecolor=c,
                       linewidths=1.6, zorder=4)
            ax.text(0.125 + 0.07 * k, y, pfmt(r.p), ha="center", va="center", fontsize=FS["small"], color=c,
                    fontweight="bold" if sig else "normal")
        ax.text(0.06, y, nlab, ha="left", va="center", fontsize=FS["small"], color=C_TEXT)
        y -= 1
    top = max(ypos.values())
    ax.text(0.06, top + 0.75, "n", fontsize=FS["small"], fontweight="bold", color=C_TEXT)
    ax.text(0.125, top + 0.75, "p id.", ha="center", fontsize=FS["small"], fontweight="bold", color=C_ID)
    ax.text(0.195, top + 0.75, "p hist.", ha="center", fontsize=FS["small"], fontweight="bold", color=C_HIST)
    ax.axvline(0, color=C_TEXT, lw=1.0)
    ax.set_xlim(-0.26, 0.05)
    ax.set_ylim(min(ypos.values()) - 0.7, top + 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Identity information lost (fraction, 95% CI)\n" + ARROWS, fontsize=FS["lab"])
    xgrid(ax)
    clean(ax, left=False)
    fig.text(0.06, 0.465, "Comparisons: a donor that does not resemble fibroblasts, and a lung-fibroblast panel",
             fontsize=FS["title"], fontweight="bold", color=C_TEXT, va="bottom")
    fig.text(0.03, 0.465, "D", fontsize=FS["letter"], fontweight="bold", va="bottom")
    save(fig, "figure_5_irradiated_clones")


# ======================================================================
# Supplementary Figure S1: the unsplit panel, every contrast, in delta-beta
# ======================================================================
SPEC = [
    ("Irradiated clones (2 Gy, ~20 doublings)", 0, "GSE112812", C1_ALL, "35 vs 11 clones"),
    ("   same, excl. 2 outlier controls", 0, "GSE112812", C1_EXCL, "35 vs 9"),
    ("Chronic low-dose radiation †", 0, "GSE197723", "chronic radiation vs untreated", "4 passage pairs"),
    ("Acute X-ray → senescence", 1, "GSE197723", "X-ray vs control", "24 pairs"),
    ("IR-induced senescence (DSF2)", 1, "GSE227160", "DSF2: IRR_SIPS vs young", "2 vs 1"),
    ("IR-induced senescence (DSF3)", 1, "GSE227160", "DSF3: IRR_SIPS vs young", "2 vs 1"),
    ("Doxorubicin senescence (DSF2)", 1, "GSE227160", "DSF2: Dox_SIPS vs young", "2 vs 1"),
    ("Doxorubicin senescence (DSF3)", 1, "GSE227160", "DSF3: Dox_SIPS vs young", "2 vs 1"),
    ("RAS oncogene-induced senescence", 2, "GSE197723", "RAS vs control", "24 pairs"),
    ("Replicative exhaustion", 2, "GSE197723", "Replication vs control", "24 pairs"),
    ("Replicative senescence (DSF2)", 2, "GSE227160", "DSF2: RS vs young", "2 vs 1"),
    ("Replicative senescence (DSF3)", 2, "GSE227160", "DSF3: RS vs young", "2 vs 1"),
    ("Progeria (HGPS) vs control", 2, "GSE149960", "HGPS vs control", "9 vs 6 people"),
]


def ylim_from(values, top_pad=0.07, bottom_pad=0.02):
    v = np.concatenate([np.atleast_1d(x) for x in values])
    return v.min() - bottom_pad, v.max() + top_pad


def fig_unsplit():
    """Every contrast on the unsplit panel, in delta-beta beyond background (+ = toward other cell types)."""
    P = "fibroblast:all"
    n_ul = int(SUMM[(SUMM.panel == "fibroblast_skin") & (SUMM.direction == "low")].n.sum())
    n_uh = int(SUMM[(SUMM.panel == "fibroblast_skin") & (SUMM.direction == "high")].n.sum())
    fs = dict(title=17, cat=13, row=14, acc=11.5, res=13.5, tick=13, axlab=13.5, legend=13, note=11.5, letter=24)
    fig = plt.figure(figsize=(16, 19), facecolor="white")
    g_top = fig.add_gridspec(1, 1, left=0.29, right=0.725, top=0.90, bottom=0.585)
    g_mid = fig.add_gridspec(1, 3, left=0.075, right=0.98, top=0.48, bottom=0.30, wspace=0.42)
    g_bot = fig.add_gridspec(1, 2, left=0.075, right=0.66, top=0.205, bottom=0.035, wspace=0.42)

    ax = fig.add_subplot(g_top[0])
    ys, bands, y, prev = [], [], 0.0, None
    for s_ in SPEC:
        cat = s_[1]
        if cat != prev:
            bands.append([cat, y, None])
            y -= 0.75
            prev = cat
        ys.append(y - 0.5)
        y -= 1.0
        bands[-1][2] = y
    y_bottom = y
    X_HEAD, X_COND, X_ACC, X_N, X_UL, X_UH = -0.675, -0.17, -0.015, 1.025, 1.255, 1.42
    tr = ax.get_yaxis_transform()
    for cat, top, bot in bands:
        ax.add_patch(plt.Rectangle((X_HEAD - 0.01, bot + 0.06), 1.57 - X_HEAD, top - bot - 0.12, transform=tr,
                                   color=CAT_FILL[cat], lw=0, zorder=0, clip_on=False))
        ax.text(X_HEAD, top - 0.4, CAT_NAMES[cat], transform=tr, ha="left", va="center", fontsize=fs["cat"],
                fontweight="bold", color=CAT_TEXT[cat])
    rows = []
    for y, (label, _, ds, contrast, nlab) in zip(ys, SPEC):
        ps = {}
        for d, col, mk, nm in DIRS:
            e = ex_row(ds, contrast, d, P)
            off = 0.15 if d == "low" else -0.15
            sig = (not np.isnan(e.p)) and e.p < 0.05
            ax.plot([0, -e.excess_shift], [y + off] * 2, color=col, lw=3.0, alpha=0.5, zorder=2)
            ax.plot(-e.excess_shift, y + off, mk, ms=11 if d == "low" else 9.5, color=col,
                    mfc=col if sig else "white", mew=2.0, zorder=4)
            ps[d] = e.p
            rows.append(dict(condition=label.strip(), data=ds, n=nlab, direction=nm, n_cpgs=int(e.n_probes),
                             shift_toward_others=-e.excess_shift, p=e.p))
        sub = label.startswith("   ")
        ax.text(X_COND, y, label.strip(), transform=tr, ha="right", va="center", color=C_TEXT,
                fontsize=fs["row"] - (1.5 if sub else 0), style="italic" if sub else "normal")
        if not sub:
            ax.text(X_ACC, y, ds, transform=tr, ha="right", va="center", fontsize=fs["acc"], color=C_MUTED)
        ax.text(X_N, y, nlab, transform=tr, ha="left", va="center", fontsize=fs["acc"] + 1, color=C_TEXT)
        if ds == "GSE227160":
            ax.text(X_UL, y, "descriptive only", transform=tr, ha="left", va="center", fontsize=fs["acc"],
                    color=C_MUTED, style="italic")
        elif np.isnan(ps["low"]):
            ax.text(X_UL, y, "too few to test", transform=tr, ha="left", va="center", fontsize=fs["acc"],
                    color=C_MUTED, style="italic")
        else:
            for (d, col, _, _), xx in zip(DIRS, (X_UL, X_UH)):
                sig = ps[d] < 0.05
                ax.text(xx, y, pfmt(ps[d]), transform=tr, ha="left", va="center", fontsize=fs["acc"] + 0.5,
                        color=col if sig else C_MUTED, fontweight="bold" if sig else "normal")
    for xx, t in ((X_N, "n"), (X_UL, "UL p"), (X_UH, "UH p")):
        ax.text(xx, 0.15, t, transform=tr, ha="left", va="bottom", fontsize=fs["acc"] + 1.5, fontweight="bold",
                color=C_TEXT)
    pd.DataFrame(rows).to_csv(R / "figure_conditions_summary.csv", index=False)
    ax.set_yticks([])
    ax.axvline(0, color=C_TEXT, lw=1.5, zorder=3)
    ax.set_xlim(-0.19, 0.13)
    ax.set_ylim(y_bottom, 0)
    ax.tick_params(axis="x", labelsize=fs["tick"])
    ax.set_xlabel("Shift beyond background CpGs, treated vs control (Δβ)\n" + ARROWS, fontsize=fs["axlab"])
    xgrid(ax)
    clean(ax, left=False)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_UL, lw=0, ms=11, label=f"UL ({n_ul:,} CpGs)"),
                       Line2D([], [], marker="o", color=C_MUTED, lw=0, ms=9, label="filled: p < 0.05"),
                       Line2D([], [], marker="D", color=C_UH, lw=0, ms=9.5, label=f"UH ({n_uh:,} CpGs)"),
                       Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=2, lw=0, ms=9,
                              label="open: p ≥ 0.05 or not testable")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=fs["legend"], frameon=False,
              columnspacing=1.6, handletextpad=0.3)
    ax.text(X_HEAD + 0.045, 1.115, "The whole, unsplit fibroblast panel across every perturbation",
            transform=ax.transAxes, fontsize=fs["title"] + 1, fontweight="bold", va="bottom")
    letter(ax, "A", X_HEAD - 0.02, 1.16, fs=fs["letter"])

    # ---- B: GSE197723 quadruplets
    ax = fig.add_subplot(g_mid[0, :2])
    quad_strips(ax, [(P, "low", C_UL, "o"), (P, "high", C_UH, "D")], (-0.2, 0.2), ylim=(-0.215, 0.03))
    ax.set_ylabel(TOWARD, fontsize=fs["axlab"])
    title(ax, "The 24 matched quadruplets (GSE197723)", fs=fs["title"], pad=46)
    letter(ax, "B", -0.1, 1.3, fs=fs["letter"])

    # ---- C: chronic radiation over passaging, unsplit
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=g_mid[0, 2], hspace=0.32)
    adj = CHRON[CHRON.panel == P].set_index("direction")
    for k, (d, col, mk, lab) in enumerate(DIRS):
        ax = fig.add_subplot(sub[k])
        g = SER[(SER.panel == P) & (SER.direction == d)]
        u = g[g.arm == "untreated"].sort_values("pdl")
        c = g[g.arm == "chronic radiation"].sort_values("pdl")
        base = u.excess.iloc[0]
        ax.plot(u.pdl, -(u.excess - base), "-o", color=C_MUTED, ms=7, lw=2.2, label="untreated")
        ax.plot(c.pdl, -(c.excess - base), "-" + mk, color=col, ms=7, lw=2.2, label="chronic radiation")
        ax.set_ylabel(f"{lab} vs PDL 3 (Δβ)", fontsize=fs["axlab"] - 1)
        ax.axhline(0, color=C_TEXT, lw=0.8, alpha=0.4)
        ygrid(ax)
        clean(ax)
        ax.text(0.03, 0.06, "same passage {:+.3f}\ndoubling-matched {:+.3f}".format(
            -adj.loc[d, "raw_shift"], -adj.loc[d, "pdl_adjusted_shift"]), transform=ax.transAxes, ha="left",
            fontsize=fs["note"] - 1, color=col, fontweight="bold")
        if k == 0:
            title(ax, "Chronic low-dose radiation\nover passaging", pad=30, fs=fs["title"])
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.98), ncol=2, fontsize=fs["note"] - 1, frameon=False,
                      handlelength=1.6, columnspacing=1.0)
            letter(ax, "C", -0.33, 1.45, fs=fs["letter"])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Population doublings", fontsize=fs["axlab"])

    # ---- D: GSE227160
    ax = fig.add_subplot(g_bot[0, 0])
    arms = [("Middle_passaged", "middle\npassage"), ("RS", "replicative\nsenesc."),
            ("IRR_SIPS", "IR\nsenesc."), ("Dox_SIPS", "doxorubicin\nsenesc.")]
    allv = []
    for i, (arm, lab) in enumerate(arms):
        for d, col, mk, _ in DIRS:
            off = -0.15 if d == "low" else 0.15
            for dn, fill in (("DSF2", True), ("DSF3", False)):
                g = per_rows("GSE227160", f"{dn}: {arm} vs young", d, P)
                v = -g[g.group == "trt"].excess_rel_ctrl.values
                allv.append(v)
                ax.scatter(np.full(len(v), i + off + (-0.045 if fill else 0.045)), v, s=70, marker=mk,
                           facecolors=col if fill else "white", edgecolors=col, linewidths=1.6, zorder=3)
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=fs["tick"] - 1)
    lo, hi = ylim_from(allv, top_pad=0.03)
    ax.set_ylim(lo - 0.01, max(hi, 0.035))
    ax.set_ylabel("Toward other cell types vs the\ndonor's young culture (Δβ)", fontsize=fs["axlab"])
    title(ax, "Two more donors (GSE227160)", fs=fs["title"])
    ygrid(ax)
    clean(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=C_UL, lw=0, ms=8, label="UL"),
                       Line2D([], [], marker="D", color=C_UH, lw=0, ms=7, label="UH"),
                       Line2D([], [], marker="o", color=C_MUTED, lw=0, ms=8, label="donor DSF2"),
                       Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.6, lw=0, ms=8,
                              label="donor DSF3")],
              loc="lower left", fontsize=fs["note"] - 1, frameon=True, edgecolor=C_GRID, ncol=2, columnspacing=0.8,
              handletextpad=0.2)
    letter(ax, "D", -0.3, 1.15, fs=fs["letter"])

    # ---- E: HGPS
    ax = fig.add_subplot(g_bot[0, 1])
    allv = []
    for j, (d, col, mk, _) in enumerate(DIRS):
        g = per_rows("GSE149960", "HGPS vs control", d, P)
        for i, grp in enumerate(("ctrl", "trt")):
            v = -g[g.group == grp].excess_rel_ctrl.values
            allv.append(v)
            strip(ax, j * 2.4 + i, v, col, marker=mk, fill=grp == "trt", s=60, w=0.4, seed=i + 3 * j)
    lo, hi = ylim_from(allv, top_pad=0.07, bottom_pad=0.03)
    for j, (d, col, mk, _) in enumerate(DIRS):
        e = ex_row("GSE149960", "HGPS vs control", d, P)
        sig = e.p < 0.05
        ax.text(j * 2.4 + 0.5, hi - 0.004, "{:+.3f}\np = {}".format(-e.excess_shift, pfmt(e.p)), ha="center",
                va="top", fontsize=fs["note"], color=col if sig else C_MUTED, fontweight="bold")
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["control\n(n=6)", "HGPS\n(n=9)", "control\n(n=6)", "HGPS\n(n=9)"], fontsize=fs["tick"] - 1)
    ax.text(0.5, -0.25, "UL", transform=ax.get_xaxis_transform(), ha="center", color=C_UL, fontweight="bold",
            fontsize=fs["row"])
    ax.text(2.9, -0.25, "UH", transform=ax.get_xaxis_transform(), ha="center", color=C_UH, fontweight="bold",
            fontsize=fs["row"])
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.set_xlim(-0.55, 3.95)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Toward other cell types\nvs control mean (Δβ)", fontsize=fs["axlab"])
    title(ax, "Progeria (GSE149960)", fs=fs["title"])
    ygrid(ax)
    clean(ax)
    letter(ax, "E", -0.3, 1.15, fs=fs["letter"])
    save(fig, "figure_S1_unsplit_panel")


# ======================================================================
# Supplementary Figure S2: irradiated clones on the unsplit panel
# ======================================================================
def fig_clones_unsplit():
    P = "fibroblast:all"
    fs = dict(title=16, res=13.5, tick=12.5, axlab=13, note=11.5, letter=24)
    raw = pd.read_csv(R / "irradiation_raw_panel_beta.csv")
    fig = plt.figure(figsize=(12, 11), facecolor="white")
    gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.42, left=0.1, right=0.98, top=0.9, bottom=0.09)

    def strip3(ax, vals, ylabel, ttl, col_mean=None):
        for i, (k, lab, col) in enumerate(CLONE_GROUPS):
            v = vals[k]
            jit = np.random.default_rng(i).uniform(-0.17, 0.17, len(v))
            out = v.index.isin(OUTLIERS)
            ax.scatter((i + jit)[~out], v.values[~out], s=60, c=col, alpha=0.85, edgecolors="white", linewidths=0.8,
                       zorder=4)
            if out.any():
                ax.scatter((i + jit)[out], v.values[out], s=85, facecolors="none", edgecolors=col, linewidths=2.2,
                           zorder=4)
            ax.plot([i - 0.28, i + 0.28], [np.mean(v)] * 2, color=col, lw=3.2, zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{lab}\n(n={len(vals[k])})" for k, lab, _ in CLONE_GROUPS], fontsize=fs["tick"])
        ax.set_ylabel(ylabel, fontsize=fs["axlab"])
        ax.set_title(ttl, fontsize=fs["title"], fontweight="bold", pad=12, loc="left")
        ygrid(ax)
        clean(ax)

    def raw_panel(ax, d, ttl):
        r = raw[raw.direction == d].set_index("sample")
        vals = {"ctrl": r[r.group == "control"].panel_mean_beta, "stable": r[r.group == "stable"].panel_mean_beta,
                "unstable": r[r.group == "unstable"].panel_mean_beta}
        nm = "UL" if d == "low" else "UH"
        strip3(ax, vals, f"Mean β over fibroblast-{nm} CpGs\n(n = {int(r.n_probes.iloc[0]):,})", ttl)
        o = vals["ctrl"][vals["ctrl"].index.isin(OUTLIERS)]
        ax.annotate("control clones\n1-1, 1-3", xy=(0.2, o.mean()), xytext=(0.3, o.mean()), fontsize=fs["note"] - 0.5,
                    color=C_MUTED, va="center")

    def excess_panel(ax, d, ttl):
        p = PER[(PER.contrast == C1_ALL) & (PER.direction == d) & (PER.panel == P)].set_index("sample")
        ctrl = p[p.group == "ctrl"].index
        stab = pd.Series({s: STAB.get(s, "NA") for s in p.index})
        vals = {"ctrl": -p.loc[ctrl, "excess_rel_ctrl"],
                "stable": -p.loc[(stab == "stable") & ~p.index.isin(ctrl), "excess_rel_ctrl"],
                "unstable": -p.loc[(stab == "unstable") & ~p.index.isin(ctrl), "excess_rel_ctrl"]}
        strip3(ax, vals, "Toward other cell types beyond\nbackground, vs control mean (Δβ)", ttl)
        ax.axhline(0, color=C_TEXT, lw=1.0, zorder=2)
        a, e = ex_row("GSE112812", C1_ALL, d, P), ex_row("GSE112812", C1_EXCL, d, P)
        u = stats.mannwhitneyu(vals["unstable"], vals["stable"], alternative="two-sided")   # computed here
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.6, hi)
        col = C_UL if d == "low" else C_UH
        for k, (t, v, p_) in enumerate((("vs all 11 controls", -a.excess_shift, a.p),
                                        ("excl. clones 1-1, 1-3", -e.excess_shift, e.p),
                                        ("unstable − stable", vals["unstable"].mean() - vals["stable"].mean(),
                                         u.pvalue))):
            sig = p_ < 0.05
            yy = 0.03 + (2 - k) * 0.09
            ax.text(0.02, yy, t, transform=ax.transAxes, fontsize=fs["note"], va="bottom", color=C_TEXT)
            ax.text(0.99, yy, "{:+.3f}  p = {}".format(v, pfmt(p_)), transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=fs["res"] - 1, color=col if sig else C_MUTED,
                    fontweight="bold" if sig else "normal")

    for (row, colm), fn, d, ttl, lt in (((0, 0), raw_panel, "low", "UL CpGs, raw β\n(lower = away from other cell types)",
                                         "A"),
                                        ((0, 1), excess_panel, "low", "UL beyond background", "B"),
                                        ((1, 0), raw_panel, "high", "UH CpGs, raw β\n(higher = away from other cell types)",
                                         "C"),
                                        ((1, 1), excess_panel, "high", "UH beyond background", "D")):
        ax = fig.add_subplot(gs[row, colm])
        fn(ax, d, ttl)
        letter(ax, lt, -0.25, 1.16, fs=fs["letter"])
    save(fig, "figure_S2_clones_unsplit")


# ======================================================================
# Supplementary Figure S3: UL and UH halves, and the background term
# ======================================================================
HALF_SPEC = [
    ("Acute X-ray", "GSE197723", "X-ray vs control"),
    ("RAS oncogene", "GSE197723", "RAS vs control"),
    ("Replicative exhaustion", "GSE197723", "Replication vs control"),
    ("Chronic low-dose radiation", "GSE197723", "chronic radiation vs untreated"),
    ("Progeria (HGPS) vs control", "GSE149960", "HGPS vs control"),
    ("Irradiated clones, donor 1", "GSE112812", C1_ALL),
    ("   excl. 2 outlier controls", "GSE112812", C1_EXCL),
    ("Irradiated clones, donor 2", "GSE112873", C2),
]


def fig_halves():
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(1, 2, left=0.18, right=0.86, top=0.82, bottom=0.12, wspace=0.78)
    for j, lay in enumerate(("identity", "history")):
        ax = fig.add_subplot(gs[0, j])
        c = LAYER_MK[lay][0]
        ys = np.arange(len(HALF_SPEC))[::-1].astype(float)
        for yy, (lab, ds, con) in zip(ys, HALF_SPEC):
            r = ils_row(ds, con, lay)
            ax.axhspan(yy - 0.45, yy + 0.45, color="#F6F6F3" if int(yy) % 2 else "white", zorder=0)
            ax.scatter([r.frac_lost_raw], [yy + 0.22], s=40, marker="o", facecolor="white", edgecolor=C_MUTED,
                       linewidths=1.3, zorder=3)
            ax.scatter([r.frac_lost_bg], [yy + 0.22], s=40, marker="x", color=C_BG, linewidths=1.4, zorder=3)
            ax.plot([r.frac_lost] * 2, [yy - 0.38, yy + 0.38], color=c, lw=2.6, zorder=4)
            for d, mk, off in (("low", "o", -0.03), ("high", "D", -0.25)):
                e = ex_row(ds, con, d, f"fibroblast:{lay}")
                if e is None or pd.isna(e.loss_frac):
                    continue
                sig = pd.notna(e.loss_p) and e.loss_p < 0.05
                ax.scatter([e.loss_frac], [yy + off], s=46, marker=mk, facecolor=c if sig else "white", edgecolor=c,
                           linewidths=1.4, zorder=5)
            if j == 0:
                ax.text(-0.02, yy, lab.strip(), transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=FS["note"], color=C_TEXT, style="italic" if lab.startswith(" ") else "normal")
            pl, ph = r.p_low, r.p_high
            ax.text(1.03, yy, "{:.0%}".format(r.jeffreys_share_low), transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=FS["small"], color=C_TEXT)
            ax.text(1.2, yy, pfmt(pl) if pd.notna(pl) else "—", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=FS["small"], color=c)
            ax.text(1.47, yy, pfmt(ph) if pd.notna(ph) else "—", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=FS["small"], color=c)
        for xx, t in ((1.03, "UL %"), (1.2, "p UL"), (1.47, "p UH")):
            ax.text(xx, len(HALF_SPEC) - 0.35, t, transform=ax.get_yaxis_transform(), ha="left", fontsize=FS["small"],
                    fontweight="bold", color=C_TEXT)
        ax.axvline(0, color=C_TEXT, lw=1.0, zorder=2)
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(HALF_SPEC) - 0.1)
        ax.set_xlabel("Information lost (fraction)\n" + ARROWS, fontsize=FS["lab"])
        xgrid(ax)
        clean(ax, left=False)
        title(ax, "Identity CpGs" if lay == "identity" else "Proliferation-history CpGs", pad=28)
        letter(ax, "AB"[j], -0.08 if j else -0.62, 1.12)
    fig.legend(handles=[Line2D([], [], color=C_TEXT, lw=2.6, label="combined (net), as in Figures 3 and 4"),
                        Line2D([], [], marker="o", color=C_TEXT, lw=0, ms=8, label="UL half (net)"),
                        Line2D([], [], marker="D", color=C_TEXT, lw=0, ms=7, label="UH half (net)"),
                        Line2D([], [], marker="o", color=C_MUTED, mfc="white", lw=0, ms=8,
                               label="combined, panel's own shift"),
                        Line2D([], [], marker="x", color=C_BG, lw=0, ms=8, mew=1.4, label="combined, background term"),
                        Line2D([], [], marker="o", color="#555555", mfc="white", lw=0, ms=8,
                               label="open half: p ≥ 0.05 or untested")],
               loc="upper left", bbox_to_anchor=(0.2, 0.975), ncol=3, frameon=False, fontsize=FS["note"],
               columnspacing=1.6)
    save(fig, "figure_S3_ul_uh_background")


# ======================================================================
# Supplementary Figures S4-S5: skin
# ======================================================================
SKIN_FULL = {"fibroblast": "Fibroblast", "keratinocyte": "Keratinocyte", "endothelial": "Endothelial"}


def fig_skin():
    fs = dict(title=17, res=13.5, tick=13, axlab=13, note=11.5, letter=24)
    items = [(p, d, col, mk, nm) for p in SKIN_FULL for d, col, mk, nm in DIRS]
    xpos = [c * 2.5 + j * 1.1 for c in range(3) for j in range(2)]
    nprobe = {(p, d): int(VAL[(VAL.panel == f"{p}:all") & (VAL.direction == d)].n_probes.iloc[0])
              for p, d, *_ in items}
    fig = plt.figure(figsize=(19, 14), facecolor="white")
    g_top = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.15, 1.15], wspace=0.3, left=0.055, right=0.985,
                             top=0.88, bottom=0.57)
    g_bot = fig.add_gridspec(1, 3, wspace=0.42, left=0.055, right=0.985, top=0.43, bottom=0.08)

    def xaxis_groups(ax, labels, y=-0.1, gfs=fs["tick"], pad=0.6):
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=fs["tick"])
        for t, (_, _, col, _, _) in zip(ax.get_xticklabels(), items):
            t.set_color(col)
            t.set_fontweight("bold")
        for c, p in enumerate(SKIN_FULL):
            ax.text(c * 2.5 + 0.55, y, SKIN_FULL[p], transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=gfs, fontweight="bold", color=C_TEXT)
        ax.set_xlim(-pad, xpos[-1] + pad)

    ax = fig.add_subplot(g_top[0])
    for x, (p, d, col, mk, nm) in zip(xpos, items):
        v = VAL[(VAL.panel == f"{p}:all") & (VAL.direction == d)].set_index("layer").mean_beta
        ax.bar(x - 0.21, v["dermis"], 0.4, color=col, zorder=3)
        ax.bar(x + 0.21, v["epidermis"], 0.4, color=col, alpha=0.4, hatch="///", edgecolor="white", zorder=3)
    xaxis_groups(ax, [nm for *_, nm in items], gfs=fs["tick"] - 1.5, pad=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean β in sun-protected skin", fontsize=fs["axlab"])
    ax.set_title("Panels validate by layer", fontsize=fs["title"], fontweight="bold", loc="left", pad=12)
    ax.legend(handles=[Patch(color=C_MUTED, label="dermis"),
                       Patch(facecolor=C_MUTED, alpha=0.4, hatch="///", edgecolor="white", label="epidermis")],
              loc="upper right", fontsize=fs["note"], frameon=True, edgecolor=C_GRID, ncol=2, columnspacing=1)
    ygrid(ax)
    clean(ax)
    letter(ax, "A", -0.22, 1.14, fs=fs["letter"])

    def layer_vals(layer, p, d):
        g = per_rows("GSE51954", f"sun exposed vs protected, {layer}", d, f"{p}:all", layer=layer)
        return g[g.age != "all"]

    allv = np.concatenate([-layer_vals(l_, p, d).excess_diff.values for l_ in ("epidermis", "dermis")
                           for p, d, *_ in items])
    lo, hi = allv.min() - 0.03, allv.max() + 0.04
    for k, layer in enumerate(("epidermis", "dermis")):
        ax = fig.add_subplot(g_top[1 + k])
        for j, (x, (p, d, col, mk, nm)) in enumerate(zip(xpos, items)):
            g = layer_vals(layer, p, d)
            for age, fill in (("young", True), ("old", False)):
                v = -g[g.age == age].excess_diff.values
                xx = x + (-0.15 if fill else 0.15) + np.random.default_rng(j * 10 + int(fill)).uniform(-0.06, 0.06,
                                                                                                       len(v))
                ax.scatter(xx, v, s=50, marker=mk, facecolors=col if fill else "white", edgecolors=col,
                           linewidths=1.4, alpha=0.85, zorder=3)
            e = ex_row("GSE51954", f"sun exposed vs protected, {layer}", d, f"{p}:all", layer=layer, age="all")
            ax.plot([x - 0.34, x + 0.34], [-e.excess_shift] * 2, color=col, lw=3.2, zorder=4)
            sig = e.p < 0.05
            ax.text(x, hi - 0.003, "{:+.3f}".format(-e.excess_shift), ha="center", va="top", fontsize=fs["res"] - 1,
                    color=col if sig else C_MUTED, fontweight="bold" if sig else "normal")
            ax.text(x, hi - 0.016, pfmt(e.p), ha="center", va="top", fontsize=fs["note"] - 3,
                    color=col if sig else C_MUTED)
        ax.axhline(0, color=C_TEXT, lw=1.0)
        ax.axhspan(-0.01, 0.01, color="#EDEDE8", zorder=0)
        ax.set_ylim(lo, hi)
        xaxis_groups(ax, [nm for *_, nm in items])
        if k == 0:
            ax.set_ylabel("Toward other cell types, sun-exposed vs\nprotected (same person), beyond background (Δβ)",
                          fontsize=fs["axlab"])
        npairs = int(ex_row("GSE51954", f"sun exposed vs protected, {layer}", "low", "fibroblast:all",
                            layer=layer, age="all").n)
        ax.set_title(f"{layer.capitalize()}: sun exposure ({npairs} people)", fontsize=fs["title"],
                     fontweight="bold", loc="left", pad=12)
        ygrid(ax)
        clean(ax)
        if k == 1:
            ax.legend(handles=[Line2D([], [], marker="o", color=C_MUTED, lw=0, ms=8, label="young (20–34 y)"),
                               Line2D([], [], marker="o", color=C_MUTED, mfc="white", mew=1.5, lw=0, ms=8,
                                      label="old (65–90+ y)"),
                               Line2D([], [], color=C_TEXT, lw=3.2, label="mean"),
                               Patch(color="#EDEDE8", label="|Δ| < 0.01: negligible")],
                      loc="lower right", fontsize=fs["note"] - 1, frameon=True, edgecolor=C_GRID, ncol=2,
                      columnspacing=1, handletextpad=0.4)
        letter(ax, "BC"[k], -0.2 if k == 0 else -0.12, 1.14, fs=fs["letter"])

    g = PER[(PER.dataset == "GSE51954") & (PER.layer == "dermis") & (PER.age != "all")]
    w = g.pivot_table(index="trt", columns=["panel", "direction"], values="excess_diff")
    xv = w[("keratinocyte:all", "low")]

    def carry(y):                                                   # computed here
        rho, p_rho = stats.spearmanr(xv, y)
        r = stats.linregress(xv, y)
        p_int = 2 * stats.t.sf(abs(r.intercept / r.intercept_stderr), len(xv) - 2)
        return rho, p_rho, r, p_int

    for k, d in enumerate(("low", "high")):
        ax = fig.add_subplot(g_bot[k])
        yv = -w[("fibroblast:all", d)]
        col = C_UL if d == "low" else C_UH
        nm = "UL" if d == "low" else "UH"
        ax.scatter(xv, yv, s=70, color=col, edgecolors="white", linewidths=0.8, zorder=3)
        rho, p_rho, r, p_int = carry(yv)
        xx = np.linspace(min(xv.min(), 0), xv.max(), 10)
        ax.plot(xx, r.intercept + r.slope * xx, color=C_MUTED, lw=2, ls="--")
        ax.plot(0, r.intercept, "o", ms=10, mfc="white", mec=C_TEXT, mew=2, zorder=5)
        ax.axhline(0, color=C_TEXT, lw=0.8, alpha=0.5)
        ax.axvline(0, color=C_TEXT, lw=0.8, alpha=0.5)
        ax.set_xlabel("Keratinocyte signal in dermis\n(keratinocyte UL panel; + = more keratinocyte-like)",
                      fontsize=fs["axlab"])
        ax.set_ylabel(f"Fibroblast {nm} in dermis,\ntoward other cell types (Δβ)", fontsize=fs["axlab"])
        ax.text(0.03, 0.97, "Spearman ρ = {:.2f}\np = {}".format(rho, pfmt(p_rho)), transform=ax.transAxes,
                ha="left", va="top", fontsize=fs["res"], color=col if p_rho < 0.05 else C_MUTED, fontweight="bold")
        ax.text(0.97, 0.04, "○ at zero carry-over: {:+.3f} (p = {})".format(r.intercept, pfmt(p_int)),
                transform=ax.transAxes, va="bottom", ha="right", fontsize=fs["note"], color=C_TEXT,
                bbox=dict(facecolor="white", edgecolor=C_GRID, boxstyle="round,pad=0.3"))
        ygrid(ax)
        clean(ax)
        ax.set_title(f"Carry-over check: fibroblast {nm}", fontsize=fs["title"], fontweight="bold", loc="left",
                     pad=12)
        letter(ax, "DE"[k], -0.22, 1.14, fs=fs["letter"])
    en = {d: carry(-w[("endothelial:all", d)]) for d in ("low", "high")}
    print("endothelial carry-over: " + "; ".join("{} rho {:.2f}, at zero {:+.3f} (p = {})".format(
        d, en[d][0], en[d][2].intercept, pfmt(en[d][3])) for d in en))

    ax = fig.add_subplot(g_bot[2])
    for j, (d, col, mk, nm) in enumerate(DIRS):
        v = -per_rows("GSE197723", "keratinocyte chronic radiation vs untreated", d, "keratinocyte:all").excess_diff
        ax.scatter(np.full(len(v), j) + np.linspace(-0.1, 0.1, len(v)), v, s=80, marker=mk, color=col,
                   edgecolors="white", zorder=3)
        ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color=col, lw=3.2, zorder=4)
        ax.text(j, 0.0072, "{:+.3f}".format(v.mean()), ha="center", va="top", fontsize=fs["res"], color=C_MUTED,
                fontweight="bold")
    ax.axhline(0, color=C_TEXT, lw=1.0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.008, 0.008)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Keratinocyte UL\n({nprobe[('keratinocyte', 'low')]:,} CpGs)",
                        f"Keratinocyte UH\n({nprobe[('keratinocyte', 'high')]:,} CpGs)"], fontsize=fs["tick"])
    ax.set_ylabel("Toward other cell types, chronic radiation\nvs untreated (doubling-matched, Δβ)",
                  fontsize=fs["axlab"])
    ax.set_title("Keratinocytes, chronic radiation\n(GSE197723)", fontsize=fs["title"], fontweight="bold", loc="left",
                 pad=12)
    ygrid(ax)
    clean(ax)
    letter(ax, "F", -0.25, 1.22, fs=fs["letter"])
    save(fig, "figure_S4_skin_celltype_panels")


PROBE_ITEMS = [(p, d, col, nm) for p in ("fibroblast", "keratinocyte") for d, col, _, nm in DIRS]


def probe_axes(ax, k, layer, lim, summ, fs, xpos):
    ax.axhline(0, color=C_TEXT, lw=1.2, zorder=5)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(xpos[0] - 0.75, xpos[-1] + 0.75)
    ax.set_xticks(xpos)
    ax.set_xticklabels([nm for *_, nm in PROBE_ITEMS], fontsize=fs["tick"])
    for t, (_, _, col, _) in zip(ax.get_xticklabels(), PROBE_ITEMS):
        t.set_color(col)
        t.set_fontweight("bold")
    for c, p in enumerate(("fibroblast", "keratinocyte")):
        ax.text(c * 2.0 + 0.425, -0.075, SKIN_FULL[p], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=fs["tick"], fontweight="bold", color=C_TEXT)
    n = int(summ[summ.layer == layer].n_pairs.iloc[0])
    ax.set_title("{}: sun exposure ({} people)".format(layer.capitalize(), n), fontsize=fs["title"],
                 fontweight="bold", loc="left", pad=12)
    if k == 0:
        ax.set_ylabel("Per-CpG shift toward other cell types,\nsun-exposed vs protected (mean over people, Δβ)",
                      fontsize=fs["axlab"])
    ygrid(ax)
    clean(ax)
    letter(ax, "AB"[k], -0.16 if k == 0 else -0.08, 1.14, fs=fs["letter"])
    if k == 1:
        for y0, y1, lab in ((0.54, 0.95, "toward other cell types"), (0.46, 0.05, "away from them")):
            ax.annotate("", xy=(1.045, y1), xytext=(1.045, y0), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=2))
            ax.text(1.085, (y0 + y1) / 2, lab, transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=fs["note"], color=C_MUTED, fontweight="bold")


def probe_labels(ax, x, s, col, lim, fs, top_gap):
    sig = s.mannwhitney_p < 0.05 and abs(s.excess) >= 0.005
    ax.text(x, lim * 0.98, "{:+.3f}".format(-s.excess), ha="center", va="top", fontsize=fs["res"],
            color=col if sig else C_MUTED, fontweight="bold", zorder=8)
    ax.text(x, lim * 0.98 - top_gap, ptext(s.mannwhitney_p), ha="center", va="top",
            fontsize=fs["note"] - 1, color=col if sig else C_MUTED, zorder=7)
    ax.text(x, -lim * 0.98, "{:.0f}%  vs  {:.0f}%".format(100 * s.frac_collapse, 100 * s.bg_frac_collapse),
            ha="center", va="bottom", fontsize=fs["note"] - 1, color=C_MUTED, zorder=7)


def fig_skin_probe_level():
    fs = dict(title=18, res=14.5, tick=14, axlab=14, note=12.5, letter=26)
    dl = pd.read_csv(R / "skin_probe_level_deltas.csv")
    dens = pd.read_csv(R / "skin_probe_level_density.csv")
    summ = pd.read_csv(R / "skin_probe_level_summary.csv")
    xpos = [c * 2.0 + j * 0.85 for c in range(2) for j in range(2)]
    lim = max(abs(np.percentile(dl.signed_delta, [0.3, 99.7]))) * 1.08
    half = 0.36
    fig = plt.figure(figsize=(17, 9), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.16, left=0.075, right=0.985, top=0.86, bottom=0.17)
    for k, layer in enumerate(("epidermis", "dermis")):
        ax = fig.add_subplot(gs[k])
        for x, (p, d, col, nm) in zip(xpos, PROBE_ITEMS):
            s = summ[(summ.layer == layer) & (summ.panel == p) & (summ.direction == d)].iloc[0]
            q = dens[(dens.layer == layer) & (dens.panel == p) & (dens.direction == d)]
            w = {lab: g.set_index("x").density for lab, g in q.groupby("set")}
            scale = half / max(v.max() for v in w.values())
            for lab, side, fc, ec in (("background", -1, "#E4E4E0", C_MUTED), ("panel", 1, col, "white")):
                v = w[lab]
                ax.fill_betweenx(-v.index.to_numpy(), x, x + side * v.values * scale, facecolor=fc, edgecolor=ec,
                                 lw=1.1, alpha=0.95 if lab == "panel" else 1, zorder=3)
            for m, side in ((-s.bg_mean_signed, -1), (-s.mean_signed, 1)):
                ax.plot([x, x + side * half * 0.75], [m] * 2, color=C_TEXT, lw=2.2, zorder=6)
            probe_labels(ax, x, s, col, lim, fs, 0.013)
        probe_axes(ax, k, layer, lim, summ, fs, xpos)
    fig.legend(handles=[Patch(facecolor="#E4E4E0", edgecolor=C_MUTED, label="matched background CpGs"),
                        Patch(facecolor=C_MUTED, edgecolor="white", label="panel CpGs"),
                        Line2D([], [], color=C_TEXT, lw=2.2, label="mean")],
               loc="lower center", bbox_to_anchor=(0.53, 0.005), ncol=3, fontsize=fs["note"], frameon=True,
               edgecolor=C_GRID, handletextpad=0.6, columnspacing=2)
    save(fig, "figure_S5_skin_probe_level")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    fig_concept()
    fig_two_layers()
    fig_information()
    fig_all()
    fig_clones()
    fig_unsplit()
    fig_clones_unsplit()
    fig_halves()
    fig_skin()
    fig_skin_probe_level()
    print(f"figures written to {OUT}")
