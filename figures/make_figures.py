import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT_DIR = (os.path.join(ROOT, "paper", "figures")
           if os.path.isdir(os.path.join(ROOT, "paper", "figures")) else HERE)
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)

SURVEYS = (
    ("documents", "backbone_survey.json"),
    ("shapes", "shape_survey.json"),
    ("real scans", "funsd_scan_survey.json"),
    ("large scans", "docvqa_scan_survey.json"),
    ("arxiv", "real_scan_wide_survey.json"),
    ("identity documents", "midv_full_survey.json"),
    ("domain SAE", "domain_sae_survey.json"),
)

COLORS = {
    "documents": "#1f77b4",
    "shapes": "#d62728",
    "real scans": "#2ca02c",
    "large scans": "#8c564b",
    "arxiv": "#9467bd",
    "identity documents": "#e377c2",
    "real captures": "#ff7f0e",
    "domain SAE": "#7f7f7f",
    "sweep": "#000000",
}


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as handle:
        return json.load(handle)


def points():

    rows = []
    for domain, path in SURVEYS:
        for encoder, row in load(path).items():
            if "ratio" not in row:
                continue
            rows.append({
                "domain": domain,
                "encoder": encoder,
                "spread": row["per_dim_ratio_spread"],
                "headroom": (row["ratio_after_diagonal"] / row["ratio"] - 1.0) * 100.0,
                "ratio": row["ratio"],
                "recall": row["NN_R@1"],
            })
    return rows


def figure_bound():
    rows = points()
    sweep = [{"domain": "sweep", "spread": row["spread"],
              "headroom": row["oracle_diagonal"]}
             for level in load("strength_sweep.json").values()
             for row in level.values()]
    synthetic = load("bound_validation.json")["sigma_sweep"]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for domain, _ in SURVEYS:
        subset = [r for r in rows if r["domain"] == domain]
        if not subset:
            continue
        ax.scatter([r["spread"] for r in subset], [r["headroom"] for r in subset],
                   s=34, color=COLORS[domain], label=domain, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.scatter([r["spread"] for r in sweep], [r["headroom"] for r in sweep],
               marker="x", s=30, color=COLORS["sweep"], label="layout-strength sweep",
               zorder=3)
    xs = [v["spread"] for v in synthetic.values()]
    ys = [v["headroom"] * 100.0 for v in synthetic.values()]
    order = np.argsort(xs)
    ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], color="black",
            linestyle="--", linewidth=1.2, label="synthetic model")
    ax.axvline(1.7 / np.sqrt(6), color="grey", linestyle=":", linewidth=1.2)
    ax.annotate("estimator floor\n(6 nuisance realisations)", xy=(0.69, 60),
                xytext=(0.30, 300), fontsize=7, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", linewidth=0.8))
    ax.set_xlabel("spread of the per-dimension content/nuisance ratio")
    ax.set_ylabel("oracle diagonal headroom  (%)")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.3)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_bound.pdf"))
    plt.close(fig)


def figure_teaser():
    from experiment_utils import load as load_domain
    from factor_eval import variance_decomposition

    fig = plt.figure(figsize=(7.0, 2.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    for encoder, colour in (("clip_openai", "#1f77b4"), ("dinov2_l", "#d62728")):
        features, contents, layouts = load_domain("documents", encoder)
        layout_var, content_var = variance_decomposition(features, contents, layouts)
        per_dim = (content_var / (layout_var + 1e-12)).numpy()
        per_dim = per_dim / np.median(per_dim)
        ax.plot(np.sort(per_dim), color=colour, linewidth=1.4,
                label=f"{encoder}  spread {per_dim.std() / per_dim.mean():.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("dimension, sorted", fontsize=7)
    ax.set_ylabel("$c_d / v_d$, normalised", fontsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=6)

    rows = points()
    sweep = [{"domain": "sweep", "spread": row["spread"], "headroom": row["oracle_diagonal"]}
             for level in load("strength_sweep.json").values() for row in level.values()]
    ax2 = fig.add_subplot(grid[0, 1])
    for domain, _ in SURVEYS:
        subset = [r for r in rows if r["domain"] == domain]
        if subset:
            ax2.scatter([r["spread"] for r in subset], [r["headroom"] for r in subset],
                        s=26, color=COLORS[domain], label=domain, edgecolor="white",
                        linewidth=0.4, zorder=3)
    ax2.scatter([r["spread"] for r in sweep], [r["headroom"] for r in sweep], marker="x",
                s=22, color="black", label="layout-strength sweep", zorder=3)
    synthetic = load("bound_validation.json")["sigma_sweep"]
    xs = np.array([v["spread"] for v in synthetic.values()])
    ys = np.array([v["headroom"] * 100 for v in synthetic.values()])
    order = np.argsort(xs)
    ax2.plot(xs[order], ys[order], color="black", linestyle="--", linewidth=1.0,
             label="synthetic model")
    ax2.axvspan(0, 0.22, color="#1f77b4", alpha=0.07)
    ax2.axvspan(0.40, 1.3, color="#d62728", alpha=0.07)
    ax2.axvline(1.7 / np.sqrt(6), color="grey", linestyle=":", linewidth=1.0)
    ax2.set_yscale("log")
    ax2.set_xlim(0, 1.3)
    ax2.set_xlabel("spread of $c_d / v_d$", fontsize=7)
    ax2.set_ylabel("headroom of a per-dimension rescaling  (%)", fontsize=7)
    ax2.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left")
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_teaser.pdf"))
    plt.close(fig)


def figure_geometry():

    from factor_eval import variance_decomposition
    from experiment_utils import load as load_domain

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, encoder in zip(axes[:2], ("clip_openai", "dinov2_l")):
        features, contents, layouts = load_domain("documents", encoder)
        layout_var, content_var = variance_decomposition(features, contents, layouts)
        x = np.maximum(layout_var.numpy(), 1e-9)
        y = np.maximum(content_var.numpy(), 1e-9)
        ax.scatter(x, y, s=5, alpha=0.5, color="#1f77b4" if encoder == "clip_openai" else "#d62728")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=0.8)
        per_dim = y / x
        ax.set_title(f"{encoder}\nspread={per_dim.std() / per_dim.mean():.3f}", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("nuisance variance $v_d$", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("content variance $c_d$", fontsize=7)

    rows = [r for r in points() if r["encoder"] == "clip_openai"]
    ax = axes[2]
    names = [r["domain"] for r in rows]
    ax.bar(range(len(rows)), [r["spread"] for r in rows], color="#1f77b4", width=0.6)
    ax.axhline(1.7 / np.sqrt(6), color="grey", linestyle=":", linewidth=1.0)
    ax.text(0.05, 1.7 / np.sqrt(6) + 0.02, "floor, L=6", fontsize=6, color="grey")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6)
    ax.set_ylabel("spread", fontsize=7)
    ax.set_title("clip_openai", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_geometry.pdf"))
    plt.close(fig)


def figure_strength():
    data = load("strength_sweep.json")
    levels = sorted(data)
    encoders = sorted(next(iter(data.values())))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for encoder in encoders:
        ratios = [data[l][encoder]["ratio"] for l in levels]
        spreads = [data[l][encoder]["spread"] for l in levels]
        heads = [data[l][encoder]["oracle_diagonal"] for l in levels]
        axes[0].plot(ratios, spreads, marker="o", markersize=3, label=encoder)
        axes[1].plot(ratios, heads, marker="s", markersize=3, label=encoder)
    axes[0].set_xlabel("content/nuisance ratio (falls as the nuisance grows)")
    axes[0].set_ylabel("spread of the per-dimension ratio")
    axes[0].invert_xaxis()
    axes[1].set_xlabel("content/nuisance ratio")
    axes[1].set_ylabel("oracle diagonal headroom (%)")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_strength.pdf"))
    plt.close(fig)


def figure_rank_calibration():
    rank = load("rank_sweep.json")
    pair = load("pair_curve.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for key, row in rank.items():
        ranks = [int(k.split("=")[1]) for k in row]
        recalls = [row[k]["NN_R@1"] for k in row]
        axes[0].plot(ranks, recalls, marker="o", markersize=3, label=key)
    axes[0].set_xlabel("rank of the removed subspace")
    axes[0].set_ylabel("NN_R@1")
    axes[0].legend(fontsize=5, frameon=False)

    sizes = sorted(int(k) for k in pair["sizes"])
    axes[1].plot(sizes, [pair["sizes"][str(s)]["NN_R@1"] for s in sizes],
                 marker="o", markersize=3)
    axes[1].axhline(pair["baseline"]["NN_R@1"], color="grey", linestyle="--",
                    linewidth=1.0)
    axes[1].text(sizes[0], pair["baseline"]["NN_R@1"] + 0.004, "raw", fontsize=6,
                 color="grey")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("calibration contents")
    axes[1].set_ylabel("NN_R@1")

    noise = sorted(float(k) for k in pair["noise"])
    axes[2].plot(noise, [pair["noise"][f"{n:.2f}"]["NN_R@1"] for n in noise],
                 marker="o", markersize=3)
    axes[2].axhline(pair["baseline"]["NN_R@1"], color="grey", linestyle="--",
                    linewidth=1.0)
    axes[2].set_xlabel("fraction of mismatched pairs")
    axes[2].set_ylabel("NN_R@1")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_rank_calibration.pdf"))
    plt.close(fig)


def figure_budget():
    released = load("topk_budget.json")
    scaled = load("domain_sae_large.json")["budgets"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    keys = ["k=16", "k=32", "k=64", "k=128", "k=256", "k=1024"]
    xs = [16, 32, 64, 128, 256, 1024]
    for domain, row in released.items():
        if domain != "documents":
            continue
        ax.plot(xs, [row[k]["NN_R@1"] for k in keys], marker="o", markersize=3,
                label=f"released SAE, {domain}")
        ax.axhline(row["dense"]["NN_R@1"], color="grey", linestyle=":",
                   linewidth=0.8)
    ax.plot(xs, [scaled[k]["NN_R@1"] for k in keys], marker="s", markersize=3,
            label="in-domain SAE, 24k images")
    ax.axhline(scaled["dense"]["NN_R@1"], color="black", linestyle=":",
               linewidth=0.8)
    ax.axvline(16.8, color="red", linestyle="--", linewidth=0.9)
    ax.text(17, 0.35, "17 of 64 slots", fontsize=6, color="red", rotation=90)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("top-$k$ budget for the content stream")
    ax.set_ylabel("NN_R@1")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_budget.pdf"))
    plt.close(fig)


def figure_prediction():
    data = load("bound_prediction.json") or {}
    rows = data.get("rows", [])
    if not rows:
        return
    xs = np.array([max(r["predicted"], 1e-3) for r in rows])
    ys = np.array([max(r["measured"], 1e-3) for r in rows])
    domains = sorted({r["domain"] for r in rows})
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for domain in domains:
        sel = [i for i, r in enumerate(rows) if r["domain"] == domain]
        ax.scatter(xs[sel], ys[sel], s=26, label=domain,
                   color=COLORS.get(domain, "black"), edgecolor="white",
                   linewidth=0.4, zorder=3)
    lims = [min(xs.min(), ys.min()) * 0.6, max(xs.max(), ys.max()) * 1.6]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=0.8,
            label="idealised bound = measurement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("idealised headroom from the proposition (%)", fontsize=7)
    ax.set_ylabel("measured oracle headroom (%)", fontsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_prediction.pdf"))
    plt.close(fig)


def figure_method_slot():


    path = os.path.join(OUT_DIR, "fig6_method.pdf")
    if os.path.exists(path):
        return
    fig = plt.figure(figsize=(3.4, 2.1))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.015, 0.015), 0.97, 0.97, fill=False,
                               linestyle="--", linewidth=1.0, edgecolor="#808080"))
    ax.text(0.5, 0.90, "FIGURE 6 SLOT", ha="center", va="center",
            fontsize=9, color="#404040")
    ax.text(0.5, 0.75, "the method overview, to be drawn", ha="center",
            va="center", fontsize=7, style="italic", color="#404040")
    ax.text(0.5, 0.55, "left: two renderings of one content, their difference, the top-r subspace",
            ha="center", va="center", fontsize=5.6, color="#404040")
    ax.text(0.5, 0.43, "right: x  ->  x - B B^T x, one matrix product at inference",
            ha="center", va="center", fontsize=5.6, color="#404040")
    ax.text(0.5, 0.24, "overwrite figures/fig6_method.pdf with the drawing", ha="center",
            va="center", fontsize=5.6, color="#404040")
    ax.text(0.5, 0.14, "keep the aspect ratio, or change the width in the tex",
            ha="center", va="center", fontsize=5.6, color="#404040")
    fig.savefig(path)
    plt.close(fig)


def main():
    figure_geometry()
    figure_teaser()
    figure_bound()
    figure_strength()
    figure_rank_calibration()
    figure_budget()
    figure_prediction()
    figure_method_slot()
    for name in sorted(os.listdir(OUT_DIR)):
        if name.endswith(".pdf"):
            print("wrote", os.path.relpath(os.path.join(OUT_DIR, name)))


if __name__ == "__main__":
    main()
