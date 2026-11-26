"""One figure of the main model results, read from data/results/ and data/predictions/ (nothing is recomputed). The models
shown are those in results_metrics.csv; axis limits and counts come from the data. Run from the Model Analysis folder,
after evaluate.py and spatial.py evaluate (both for the same models):

    python evaluation/plot_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predictions as P

RESULTS = P.EVAL_ROOT / "data" / "results"
OUT = P.EVAL_ROOT.parent / "Project Context" / "Blue Carbon Mapping" / "baseline_results.png"
# name in the results files -> (label, colour); the colour follows the model, whichever models are shown
STYLE = {"null": ("Null (training mean)", "#8a8a8a"), "rf": ("Random Forest", "#257de0"), "xgb": ("XGBoost", "#f47230"),
         "attunet": ("Attention U-Net", "#00b894"), "unetpp": ("UNet++", "#6a4ce2")}
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def read(name):
    return pd.read_csv(RESULTS / f"results_{name}.csv", keep_default_na=False, na_values=[""])    # keeps the name "null"


def style(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=34)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.2, color=MUTED, va="bottom", linespacing=1.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=9)


def offsets(n, width=0.72):
    """Positions of n marks side by side around 0, and the room each one has."""
    step = width / n
    return (np.arange(n) - (n - 1) / 2) * step, step


def overall(ax, models):
    test, spatial, floor = read("metrics"), read("spatial_metrics"), read("noise_floor").iloc[0]
    same = read("spatial_vs_random").set_index("model")
    dx, _ = offsets(len(models), 0.66)
    values = []
    for x, table in enumerate([test, spatial]):
        for off, m in zip(dx, models):
            r = table[(table.model == m) & (table.metric == "rmse")].iloc[0]
            ax.errorbar(x + off, r.estimate, yerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]], fmt="o", ms=8,
                        color=STYLE[m][1], capsize=3, lw=2, label=STYLE[m][0] if x == 0 else None)
            ax.annotate(f"{r.estimate:.1f}", (x + off, r.ci_hi), xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
            values += [r.ci_lo, r.ci_hi]
    # the noise floor was estimated for the TEST cells only, so it is drawn over the test group only
    ax.fill_between([-0.45, 0.45], floor.floor_ci_lo, floor.floor_ci_hi, color="#000000", alpha=0.05, lw=0)
    ax.hlines(floor.rmse_floor, -0.45, 0.45, color=MUTED, lw=1.2, ls="--")
    ax.text(0.45, floor.floor_ci_lo + 1.5, f"label-noise floor, test cells: about {floor.rmse_floor:.0f}\n"
            f"(approximate lower bound; shaded: 95% interval)", fontsize=8.3, color=MUTED, ha="right", va="bottom")
    n = [int(t[t.model == models[0]].n.iloc[0]) for t in (test, spatial)]
    ax.set_xticks([0, 1], [f"Random split\n{n[0]:,} test cells", f"Spatial hold-out\nall {n[1]:,} cells"])
    ax.set_xlim(-0.5, 1.5)
    low = np.floor((min(values + [floor.floor_ci_lo]) - 8) / 10) * 10
    ax.set_ylim(low, max(values) + 12)
    ax.set_ylabel("RMSE (Mg/ha)", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, -0.14))
    on_test = ", ".join(f"{STYLE[m][0]} {same.loc[m].rmse_spatial:.1f}" for m in models if m != "null")
    style(ax, "1. Overall error (lower is better)",
          f"Dots: estimate. Bars: 95% interval. Axis starts at {low:.0f}. The two groups use different cells.\n"
          f"Spatial model on the same test cells: {on_test}.")


def by_range(ax, models):
    t = read("by_range")
    bins = list(t[t.model == models[0]].bin)
    dx, step = offsets(len(models), 0.82)
    lo, hi = t[t.model.isin(models)].bias.min(), t[t.model.isin(models)].bias.max()
    pad = 0.09 * (hi - lo)
    for off, m in zip(dx, models):
        d = t[t.model == m].set_index("bin").loc[bins]
        ax.bar(np.arange(len(bins)) + off, d.bias, step * 0.9, color=STYLE[m][1], label=STYLE[m][0])
        if m != "null":
            for i, v in enumerate(d.bias):
                ax.annotate(f"{v:+.0f}", (i + off, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=8.3, color=INK)
    n = t[t.model == models[0]].set_index("bin").loc[bins].n
    ax.set_xticks(np.arange(len(bins)), [f"{b}\n{c} cells" for b, c in zip(bins, n)])
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed biomass (Mg/ha)", color=MUTED)
    ax.set_ylabel("Bias: predicted - observed (Mg/ha)", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax, "2. Error by biomass level (test cells)",
          "Above 0 = overestimate, below 0 = underestimate. Binning by observed value creates this slope for any model\n"
          "(regression to the mean): judge each model against the grey null bars.")


def scatter(axes, models, cells, split):
    data = {}
    for m in models:
        d = P.with_labels(P.read(m), cells, split)
        data[m] = d[d["set"] == "test"]
    top = 1.02 * max(max(d.obs.max(), d.pred.max()) for d in data.values())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#bcd7f6", "#257de0", "#0b3463"])
    marks = []
    for number, (ax, m) in enumerate(zip(axes, models), start=3):
        d = data[m]
        marks.append(ax.hexbin(d.obs, d.pred, gridsize=38, extent=(0, top, 0, top), mincnt=1, cmap=cmap))
        ax.plot([0, top], [0, top], color=INK, lw=1, ls="--")
        ax.text(0.77 * top, 0.80 * top, "perfect", rotation=45, fontsize=8.5, color=MUTED, ha="center", va="bottom",
                rotation_mode="anchor")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed (Mg/ha)", color=MUTED)
        ax.set_ylabel("Predicted (Mg/ha)", color=MUTED)
        style(ax, f"{number}. {STYLE[m][0]}", f"Predicted vs observed, test cells.\nHighest prediction: {d.pred.max():.0f}.")
    most = max(h.get_array().max() for h in marks)
    for h in marks:                                       # one colour scale for all panels
        h.set_norm(matplotlib.colors.LogNorm(vmin=1, vmax=most))
    return marks[-1]


def uncertainty(ax, models, number):
    t = read("uncertainty")
    shown = [m for m in models if m in set(t.model)]
    for m in shown:
        d = t[t.model == m].sort_values("bin")
        ax.plot(d.bin, d.mae, "-o", color=STYLE[m][1], lw=2, ms=7, label=f"{STYLE[m][0]}: mean error")
        ax.plot(d.bin, d.mean_sd, "--s", color=STYLE[m][1], lw=2, ms=6, label=f"{STYLE[m][0]}: member spread")
    bins = sorted(t.bin.unique())
    ax.set_xticks(bins, ["lowest"] + [str(b) for b in bins[1:-1]] + ["highest"])
    ax.set_ylim(0, 1.45 * t[t.model.isin(shown)].mae.max())
    ax.set_xlabel("Test cells grouped by member spread", color=MUTED)
    ax.set_ylabel("Mg/ha", color=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    flat = [STYLE[m][0] for m in shown if not t[t.model == m].mae_rises_every_bin.iloc[0]]
    note = f" {', '.join(flat)}: error does not rise in every group." if flat else ""
    style(ax, f"{number}. Member spread vs error",
          f"Spread is far smaller than the error: a relative signal, not an error bar.\n{note.strip()}")


def main():
    metrics = read("metrics")
    present = set(metrics.model)
    unknown = present - set(STYLE)
    if unknown:
        raise SystemExit(f"add a label and colour to STYLE for: {sorted(unknown)}")
    models = [m for m in STYLE if m in present]
    trained = [m for m in models if m != "null"]
    missing = present - set(read("spatial_metrics").model)
    if missing:
        raise SystemExit(f"no spatial results for {sorted(missing)}: run evaluation/spatial.py evaluate with the same models")

    cells, split = pd.read_csv(P.LABELS / "gedi_cells.csv"), P.load_split()
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    width = max(16, 4.6 * (len(trained) + 1))
    fig = plt.figure(figsize=(width, 11.8), facecolor="white")
    top = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.84, bottom=0.555, wspace=0.18, width_ratios=[1, 1.25])
    low = fig.add_gridspec(1, len(trained) + 1, left=0.055, right=0.985, top=0.39, bottom=0.12, wspace=0.28)
    overall(fig.add_subplot(top[0]), models)
    by_range(fig.add_subplot(top[1]), models)
    axes = [fig.add_subplot(low[i]) for i in range(len(trained))]
    marks = scatter(axes, trained, cells, split)
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    bar = fig.colorbar(marks, cax=fig.add_axes([left + 0.25 * (right - left), 0.05, 0.5 * (right - left), 0.011]),
                       orientation="horizontal")
    bar.set_label("Cells per hexagon (log scale)", fontsize=8.5, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    uncertainty(fig.add_subplot(low[len(trained)]), trained, len(trained) + 3)

    fig.text(0.055, 0.965, "Model results: " + ", ".join(STYLE[m][0] for m in trained), fontsize=17, fontweight="bold", color=INK)
    lines = ["Above-ground biomass at 25 m GEDI cells, Sabah mangroves."]
    paired = read("paired")
    for r in paired[(paired.metric == "rmse") & (paired.model_a != "null") & (paired.model_b != "null")].itertuples():
        verdict = "no clear winner" if r.ci_lo <= 0 <= r.ci_hi else "a difference on this split"
        lines.append(f"{STYLE[r.model_a][0]} minus {STYLE[r.model_b][0]}, test RMSE: {r.diff_estimate:+.1f} "
                     f"[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], {verdict}.")
    fig.text(0.055, 0.938, "  ".join(lines), fontsize=11, color=MUTED)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
