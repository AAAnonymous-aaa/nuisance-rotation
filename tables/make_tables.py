import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(ROOT, "outputs")
OUT_DIR = (os.path.join(ROOT, "paper", "tables")
           if os.path.isdir(os.path.join(ROOT, "paper", "tables")) else HERE)
sys.path.insert(0, ROOT)


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write(name, body):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    print("wrote", os.path.join("paper", "tables", name))


SURVEYS = (
    ("documents", "backbone_survey.json", "rendered documents"),
    ("shapes", "shape_survey.json", "shapes on textures"),
    ("real scans", "funsd_scan_survey.json", "real scans"),
    ("large scans", "docvqa_scan_survey.json", "large real scans"),
    ("arxiv", "real_scan_wide_survey.json", "arXiv first pages"),
    ("identity documents", "midv_full_survey.json", "identity documents"),
    ("domain SAE", "domain_sae_survey.json", "domain SAE heads"),
)


def all_points():
    rows = []
    for domain, path, label in SURVEYS:
        data = load(path) or {}
        for encoder, row in data.items():
            if "ratio" not in row:
                continue
            rows.append({
                "domain": domain,
                "label": label,
                "encoder": encoder,
                "ratio": row["ratio"],
                "recall": row["NN_R@1"],
                "spread": row["per_dim_ratio_spread"],
                "oracle": (row["ratio_after_diagonal"] / row["ratio"] - 1.0) * 100.0,
                "rotation": (row["ratio_after_rotation"]["4"] / row["ratio"] - 1.0) * 100.0,
            })
    return sorted(rows, key=lambda r: r["spread"])


def table_points():
    rows = all_points()
    body = ["\\begin{tabular}{llrrrrr}", "\\toprule",
            "domain & encoder & ratio & NN\\_R@1 & spread & oracle & rotation \\\\",
            "\\midrule"]
    for row in rows:
        body.append(
            f"{row['domain']} & {row['encoder'].replace('_', chr(92) + '_')} & "
            f"{row['ratio']:.2f} & {row['recall']:.3f} & {row['spread']:.3f} & "
            f"{row['oracle']:+.1f}\\% & {row['rotation']:+.1f}\\% \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_points.tex", "\n".join(body))
    return rows


def table_domains():
    from backbone_survey import load_any_grid
    grids = (
        ("rendered documents", "survey_grid", "topic $\\times$ document",
         "6-parameter layout", "synthetic"),
        ("shapes on textures", "dataset/shape_grid", "shape and colour",
         "texture, rotation, scale, offset", "synthetic"),
        ("real scans", "dataset/funsd_scan", "scanned form",
         "second scan pass (digital)", "real content"),
        ("large real scans", "dataset/docvqa_scan", "scanned page (industry archive)",
         "second scan pass (digital)", "real content"),
        ("identity documents", "dataset/midv_full", "identity document",
         "separate capture event", "real content, real nuisance"),
        ("auxiliary capture grid", "dataset/midv_capture", "identity document",
         "separate capture event", "real content, real nuisance"),
        ("arXiv first pages", "dataset/real_scan_wide", "paper (12 categories)",
         "render setting", "real content"),
        ("training grid (method)", "dataset/sae_train", "topic $\\times$ document",
         "6-parameter layout", "synthetic"),
    )
    body = ["\\begin{tabular}{llrrl}", "\\toprule",
            "domain & content axis & contents & nuisance & realisation \\\\",
            "\\midrule"]
    for name, path, content_axis, nuisance, kind in grids:
        paths, contents, layouts = load_any_grid(path)
        body.append(
            f"{name} & {content_axis} & {len(set(contents))} & "
            f"{len(set(layouts))} & {kind} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_domains_raw.tex", "\n".join(body))


def table_bands():
    rows = all_points()
    sweep = [{"spread": v["spread"], "oracle": v["oracle_diagonal"]}
             for level in (load("strength_sweep.json") or {}).values()
             for v in level.values()]
    combined = rows + sweep
    low = [r for r in combined if r["spread"] <= 0.22]
    high = [r for r in combined if r["spread"] >= 0.40]
    body = ["\\begin{tabular}{lrrr}", "\\toprule",
            "regime & points & spread & oracle headroom \\\\", "\\midrule",
            f"small spread & {len(low)} & $\\le 0.22$ & "
            f"$\\le {max(r['oracle'] for r in low):.1f}\\%$ \\\\",
            f"large spread & {len(high)} & $\\ge 0.40$ & "
            f"{min(r['oracle'] for r in high):.0f}--{max(r['oracle'] for r in high):.0f}\\% \\\\",
            "\\bottomrule", "\\end{tabular}", ""]
    write("tab_bands.tex", "\n".join(body))


def table_domain_summary():
    rows = all_points()
    body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            "domain & points & ratio & spread & oracle headroom & NN\\_R@1 \\\\",
            "\\midrule"]
    for domain, _, _ in SURVEYS:
        subset = [r for r in rows if r["domain"] == domain]
        if not subset:
            continue
        body.append(
            f"{domain} & {len(subset)} & "
            f"{min(r['ratio'] for r in subset):.2f}--{max(r['ratio'] for r in subset):.2f} & "
            f"{min(r['spread'] for r in subset):.3f}--{max(r['spread'] for r in subset):.3f} & "
            f"{min(r['oracle'] for r in subset):+.1f}--{max(r['oracle'] for r in subset):+.1f}\\% & "
            f"{min(r['recall'] for r in subset):.3f}--{max(r['recall'] for r in subset):.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_domain_summary.tex", "\n".join(body))


def table_seed_summary():
    data = load("seed_variance.json") or {}
    grouped = {}
    for key, row in data.items():
        domain = key.split("/")[0]
        grouped.setdefault(domain, []).append(row)
    body = ["\\begin{tabular}{lrrr}", "\\toprule",
            "domain & encoders & oracle sd (max) & spread rel.\\ sd (max) \\\\",
            "\\midrule"]
    for domain, rows in grouped.items():
        oracle_sd = max(r["oracle_std"] for r in rows)
        rel = max(r["spread_std"] / r["spread_mean"] for r in rows)
        body.append(f"{domain} & {len(rows)} & {oracle_sd:.1f} & {rel * 100:.1f}\\% \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_seed_summary.tex", "\n".join(body))


def table_heuristics():
    data = load("heuristics_table.json") or {}
    body = ["\\begin{tabular}{lrrr}", "\\toprule",
            "domain & IDF & masking & denoising \\\\", "\\midrule"]
    for domain, rows in data.items():
        cells = []
        for key in ("idf", "mask", "denoise"):
            values = [row[key] for row in rows.values() if key in row]
            cells.append(f"{min(values):+.1f} to {max(values):+.1f}" if values else "-")
        body.append(f"{domain} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_heuristics.tex", "\n".join(body))


def table_method():
    multi = load("projection_comparison_multi.json") or {}
    body = ["\\begin{tabular}{llrr}", "\\toprule",
            "encoder & representation & ratio & NN\\_R@1 \\\\", "\\midrule"]
    for encoder, rows in multi.items():
        order = ["raw", "closed_form_r2", "closed_form_r4", "closed_form_r8",
                 "trained_diagonal", "trained_linear", "trained_mlp"]
        for key in order:
            if key not in rows:
                continue
            name = {"raw": "raw", "trained_diagonal": "trained per-dimension",
                    "trained_linear": "trained mixing (linear)",
                    "trained_mlp": "trained mixing (MLP)"}.get(key, key.replace("_", " "))
            body.append(f"{encoder.replace('_', chr(92) + '_')} & {name} & "
                        f"{rows[key]['ratio']:.3f} & {rows[key]['NN_R@1']:.3f} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    body.append("")
    write("tab_method.tex", "\n".join(body))


def table_baselines():
    data = load("adversarial_head.json") or {}
    body = ["\\begin{tabular}{llrrr}", "\\toprule",
            "representation & supervision & ratio & NN\\_R@1 & nuisance probe \\\\",
            "\\midrule"]
    supervision = {"raw": "--", "closed form r=4": "content pairs",
                   "LEACE-style": "nuisance labels", "INLP": "nuisance labels",
                   "adversarial head": "nuisance labels"}
    for name, row in data.items():
        body.append(f"{name} & {supervision.get(name, '--')} & {row['ratio']:.2f} & "
                    f"{row['NN_R@1']:.3f} & {row['probe_accuracy']:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_baselines.tex", "\n".join(body))


def table_sae():
    released = load("sae_case_study.json") or {}
    body = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "domain & CLIP & dense head & top-$k$ code & released budget \\\\",
            "\\midrule"]
    budget = load("topk_budget.json") or {}
    for domain, rows in released.items():
        dense = rows.get("sae:logits_clip_img", {}).get("NN_R@1", float("nan"))
        sparse = rows.get("sae:sparse_codes_clip_img", {}).get("NN_R@1", float("nan"))
        clip = rows.get("clip", {}).get("NN_R@1", float("nan"))
        active = budget.get(domain, {}).get("released_mean_active")
        active_text = f"{active:.0f} of 64" if active else "--"
        body.append(f"{domain} & {clip:.3f} & {dense:.3f} & {sparse:.3f} & {active_text} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_sae.tex", "\n".join(body))


def table_sae_scale():
    scaled = load("domain_sae_large.json") or {}
    released = load("topk_budget.json", ) or {}
    body = ["\\begin{tabular}{lrr}", "\\toprule",
            "readout & ratio & NN\\_R@1 \\\\", "\\midrule"]
    order = ["k=16", "k=32", "k=64", "k=128", "k=256", "k=1024", "dense"]
    for key in order:
        row = scaled.get("budgets", {}).get(key)
        if not row:
            continue
        name = "dense (pre-selection)" if key == "dense" else f"top-$k$, $k={key.split('=')[1]}$"
        body.append(f"{name} & {row['ratio']:.2f} & {row['NN_R@1']:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_sae_scale.tex", "\n".join(body))
    dead = scaled.get("dead_latent_fraction")
    mse = scaled.get("final_train_mse")
    if dead is not None:
        write("sae_scale_facts.tex",
              f"\\newcommand{{\\saeDead}}{{{dead * 100:.1f}\\%}}\n"
              f"\\newcommand{{\\saeMse}}{{{mse:.2e}}}\n"
              f"\\newcommand{{\\saeImages}}{{{scaled.get('images')}}}\n")


def table_seed():
    data = load("seed_variance.json") or {}
    body = ["\\begin{tabular}{llrr}", "\\toprule",
            "domain & encoder & spread mean $\\pm$ sd & oracle mean $\\pm$ sd \\\\",
            "\\midrule"]
    for key, row in data.items():
        body.append(f"{key.split('/')[0]} & {key.split('/')[1].replace('_', chr(92) + '_')} & "
                    f"{row['spread_mean']:.3f} $\\pm$ {row['spread_std']:.3f} & "
                    f"{row['oracle_mean']:+.1f} $\\pm$ {row['oracle_std']:.1f}\\% \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_seed.tex", "\n".join(body))


def table_transfer():
    data = (load("subspace_controls.json") or {}).get("transfer", {})
    names = ["documents", "shapes", "real scans", "arxiv"]
    body = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "fit $\\backslash$ evaluate & documents & shapes & real scans & arXiv \\\\",
            "\\midrule"]
    for fit in names:
        if fit not in data:
            continue
        cells = []
        for ev in names:
            row = data[fit][ev]
            cells.append(f"{row['ratio']:.2f} / {row['NN_R@1']:.3f}")
        body.append(f"{fit} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_transfer.tex", "\n".join(body))


def table_protocol():
    data = load("retrieval_protocol.json") or {}
    ci = load("retrieval_ci.json") or {}
    body = ["\\begin{tabular}{llrrrr}", "\\toprule",
            "setting & & R@1 & mAP & AUROC & bootstrap $\\Delta$ R@1 \\\\",
            "\\midrule"]
    for key, row in data.items():
        interval = ci.get(key)
        ci_text = (f"{interval['delta_mean']:+.3f} "
                   f"[{interval['delta_ci95'][0]:+.3f}, {interval['delta_ci95'][1]:+.3f}]"
                   if interval else "--")
        body.append(f"{key} & raw & {row['raw']['R@1']:.3f} & {row['raw']['mAP']:.3f} & "
                    f"{row['raw']['auroc_known_vs_unknown']:.3f} & {ci_text} \\\\")
        body.append(f" & rotation & {row['rotation']['R@1']:.3f} & "
                    f"{row['rotation']['mAP']:.3f} & "
                    f"{row['rotation']['auroc_known_vs_unknown']:.3f} & \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_protocol.tex", "\n".join(body))


def table_render_controls():
    data = load("render_controls.json") or {}
    encoders = []
    for row in data.values():
        encoders = list(row)
        break
    body = ["\\begin{tabular}{l" + "r" * len(encoders) + "}", "\\toprule",
            "rendering setting & " + " & ".join(
                e.replace("_", chr(92) + "_") for e in encoders) + " \\\\", "\\midrule"]
    for name, row in data.items():
        cells = []
        for encoder in encoders:
            value = row[encoder]["ratio"]
            cells.append("n/a" if value != value else f"{value:.2f}")
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_render_controls.tex", "\n".join(body))


def table_downstream():
    data = load("downstream_tasks.json") or {}
    body = ["\\begin{tabular}{llrrrr}", "\\toprule",
            "encoder & setting & acc & NMI & ARI & NN\\_R@1 \\\\", "\\midrule"]
    for encoder, domains in data.items():
        for domain, rows in domains.items():
            label = domain if domain != "pooled" else f"pooled ({rows['contents']} contents)"
            body.append(f"{encoder.replace('_', chr(92) + '_')} & {label}, raw & "
                        f"{rows['raw']['classification']:.3f} & "
                        f"{rows['raw']['clustering']['nmi']:.3f} & "
                        f"{rows['raw']['clustering']['ari']:.3f} & "
                        f"{rows['raw']['NN_R@1']:.3f} \\\\")
            body.append(f" & {label}, rotation & "
                        f"{rows['rotation']['classification']:.3f} & "
                        f"{rows['rotation']['clustering']['nmi']:.3f} & "
                        f"{rows['rotation']['clustering']['ari']:.3f} & "
                        f"{rows['rotation']['NN_R@1']:.3f} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    body.append("")
    write("tab_downstream.tex", "\n".join(body))


def table_selection():
    data = load("selection_strategies.json") or {}
    body = ["\\begin{tabular}{llrr}", "\\toprule",
            "domain & selection rule & mean kept & NN\\_R@1 \\\\", "\\midrule"]
    for domain, row in data.items():
        if domain == "in_domain_sae_quality":
            continue
        body.append(f"{domain} & dense (no selection) & 8192 & {row['dense']['NN_R@1']:.3f} \\\\")
        body.append(f" & global top-$k$ (as released) & {row['mean_kept']['global_topk']:.0f} & "
                    f"{row['global_topk']['NN_R@1']:.3f} \\\\")
        body.append(f" & per-stream top-$k$ & {row['mean_kept']['per_stream_topk']:.0f} & "
                    f"{row['per_stream_topk']['NN_R@1']:.3f} \\\\")
        body.append(f" & value threshold & {row['mean_kept']['threshold']:.0f} & "
                    f"{row['threshold']['NN_R@1']:.3f} \\\\")
        body.append(f" & nucleus (90\\% mass) & {row['mean_kept']['nucleus']:.0f} & "
                    f"{row['nucleus']['NN_R@1']:.3f} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    body.append("")
    write("tab_selection.tex", "\n".join(body))


def table_augmentation():
    from experiment_utils import load as load_domain, metrics, project
    from experiment_utils import fit_basis
    from factor_eval import split_by_content

    data = load("augmentation_pairs.json") or {}
    body = ["\\begin{tabular}{llrr}", "\\toprule",
            "setting & pairs used at fit time & ratio & NN\\_R@1 \\\\", "\\midrule"]
    for key, row in data.items():
        domain, encoder = key.split("/")
        try:
            features, contents, layouts = load_domain(domain, encoder)
        except KeyError:

            continue
        left, _ = split_by_content(contents)
        basis = fit_basis(features, contents, layouts, rank=4, mask=left)
        matched = metrics(project(features, basis, 4), contents, layouts)
        body.append(f"{key.replace('_', chr(92) + '_')} & raw & "
                    f"{row['raw']['ratio']:.2f} & {row['raw']['NN_R@1']:.3f} \\\\")
        body.append(f" & real pairs & {matched['ratio']:.2f} & {matched['NN_R@1']:.3f} \\\\")
        body.append(f" & augmentations only & "
                    f"{row['augmentation_pairs']['ratio']:.2f} & "
                    f"{row['augmentation_pairs']['NN_R@1']:.3f} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    body.append("")
    write("tab_augmentation.tex", "\n".join(body))


def table_stats():
    data = load("retrieval_stats.json") or {}
    body = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "setting & $\\Delta$ R@1 & 95\\% interval & effect size & significant \\\\",
            "\\midrule"]
    for key, row in data.items():
        mark = "yes" if row["significant"] else "no"
        body.append(f"{key.replace('_', chr(92) + '_')} & {row['delta_mean']:+.3f} & "
                    f"[{row['delta_ci95'][0]:+.3f}, {row['delta_ci95'][1]:+.3f}] & "
                    f"{row['effect_size']:.2f} & {mark} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_stats.tex", "\n".join(body))


def table_rank_selection():
    data = load("rank_selection.json") or {}
    body = ["\\begin{tabular}{llrrrr}", "\\toprule",
            "domain & encoder & energy $r$ & held-out $r$ & fixed $r{=}4$ & held-out score \\\\",
            "\\midrule"]
    for key, row in data.items():
        domain, encoder = key.split("/")
        body.append(f"{domain} & {encoder.replace('_', chr(92) + '_')} & "
                    f"{row['energy_rank']} & {row['heldout_rank']} & "
                    f"{row['fixed_r4']:.3f} & {row['heldout_rank_score']:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_rank_selection.tex", "\n".join(body))


def table_fair():
    data = load("fair_baselines.json") or {}
    if not data:
        return
    setting = next(iter(data))
    methods = list(data[setting])
    body = ["\\begin{tabular}{llrrr}", "\\toprule",
            "setting & method & ratio & NN\\_R@1 & probe \\\\", "\\midrule"]
    for key, row in data.items():
        for index, method in enumerate(methods):
            stats = row[method]
            name = method.replace(" (", " (").replace("_", chr(92) + "_")
            first = key.replace("_", chr(92) + "_") if index == 0 else ""
            body.append(f"{first} & {name} & {stats['ratio']:.2f} & "
                        f"{stats['NN_R@1']:.3f} & {stats['probe_accuracy']:.3f} \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    body.append("")
    write("tab_fair.tex", "\n".join(body))


def table_frontier():
    data = load("diagonal_frontier.json") or {}
    if not data:
        return
    clips = list(next(iter(data.values()))["frontier"])
    body = ["\\begin{tabular}{ll" + "rr" * len(clips) + "}", "\\toprule",
            "domain & encoder & " + " & ".join(
                f"\\multicolumn{{2}}{{c}}{{clip $c={c.split('=')[1]}$}}" for c in clips) + " \\\\",
            " & & " + " & ".join("gain & R@1" for _ in clips) + " \\\\", "\\midrule"]
    for key, row in data.items():
        domain, encoder = key.split("/")
        cells = []
        for clip in clips:
            value = row["frontier"][clip]
            cells.append(f"{value['gain_percent']:+.0f}\\% & {value['recall']:.3f}")
        body.append(f"{domain} & {encoder.replace('_', chr(92) + '_')} & "
                    + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_frontier.tex", "\n".join(body))


def table_validation():
    data = load("validate_reweighting.json") or {}
    if not data:
        return
    body = ["\\begin{tabular}{llrrrrr}", "\\toprule",
            "domain & encoder & plug-in & fitted & seed sd & realisation & probe raw/fitted \\\\",
            "\\midrule"]
    for key, row in data.items():
        domain, encoder = key.split("/")
        seeds = list(row["seeds"].values())
        clip_key = next(iter(seeds[0]["optimised"]))
        gains = [s["optimised"][clip_key]["gain_percent"] for s in seeds]
        plugin = sum(s["plugin_clipped"]["gain_percent"] for s in seeds) / len(seeds)
        realisation = next(v for k, v in row["realisation_split"].items()
                           if "optimised" in k)
        probe = row["topic_probe"]
        values = [probe.get("raw"), probe.get("plugin"), probe.get("optimised")]
        degenerate = all(v in (None, 0.0) for v in values)
        probe_text = ("n/a" if degenerate else
                      f"{probe['raw']:.2f}/{probe['optimised']:.2f}")
        body.append(
            f"{domain} & {encoder.replace('_', chr(92) + '_')} & {plugin:+.1f}\\% & "
            f"{sum(gains) / len(gains):+.1f}\\% & {np.std(gains):.1f} & "
            f"{realisation:+.1f}\\% & {probe_text} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write("tab_validation.tex", "\n".join(body))


def main():
    os.makedirs(HERE, exist_ok=True)
    table_domains()
    table_points()
    table_bands()
    table_domain_summary()
    table_seed_summary()
    table_heuristics()
    table_method()
    table_baselines()
    table_sae()
    table_sae_scale()
    table_seed()
    table_transfer()
    table_protocol()
    table_render_controls()
    table_downstream()
    table_selection()
    table_augmentation()
    table_stats()
    table_rank_selection()
    table_fair()
    table_frontier()
    table_validation()


if __name__ == "__main__":
    main()
