import io
import json
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "outputs")

SECTIONS = (sorted(name for name in os.listdir(PAPER)
                   if re.match(r"\d\d_.*\.tex$", name))
            if os.path.isdir(PAPER) else [])


def read(path):
    return io.open(path, encoding="utf-8").read()


def main():
    problems = []
    have_paper = os.path.exists(os.path.join(PAPER, "main.tex"))
    text = ""
    if have_paper:
        text = "".join(read(os.path.join(PAPER, name)) for name in SECTIONS)
        text += read(os.path.join(PAPER, "main.tex"))
    else:
        print("paper sources not found: running the numeric checks only")

    for name in SECTIONS if have_paper else []:
        body = read(os.path.join(PAPER, name))
        begins = sorted(re.findall(r"\\begin\{(\w+)\}", body))
        ends = sorted(re.findall(r"\\end\{(\w+)\}", body))
        if begins != ends:
            problems.append(f"{name}: environments differ {begins} vs {ends}")
        if body.count("{") != body.count("}"):
            problems.append(f"{name}: unbalanced braces")

    inputs = re.findall(r"\\input\{([^}]*)\}", text) if have_paper else []
    for target in inputs:
        path = os.path.join(PAPER, target + ".tex")
        if not os.path.exists(path):
            problems.append(f"missing \\input: {target}")

    graphics = (re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text)
                if have_paper else [])
    for target in graphics:
        if not os.path.exists(os.path.join(PAPER, target)):
            problems.append(f"missing figure: {target}")

    allowed_environments = {
        "abstract", "equation", "itemize", "enumerate", "table", "tabular",
        "figure", "center", "algorithm", "algorithmic", "document",
    }
    for name in SECTIONS if have_paper else []:
        body = read(os.path.join(PAPER, name))
        for env in sorted(set(re.findall(r"\\begin\{(\w+)\}", body))):
            if env not in allowed_environments:
                problems.append(f"{name}: environment {env} may need a package")

    column_rule = re.compile(r"[lcrp]")
    rule_names = ("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule")
    for name in sorted(os.listdir(os.path.join(PAPER, "tables"))):
        if not name.endswith(".tex"):
            continue
        body = read(os.path.join(PAPER, "tables", name))
        match = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body)
        if not match:
            continue
        n_columns = len(column_rule.findall(match.group(1)))
        rows = body[match.end():].split("\\end{tabular}")[0].split("\\\\")
        for index, row in enumerate(rows):
            row = row.strip()
            if not row or any(row.startswith(rule) for rule in rule_names):
                continue
            cells = len(row.split("&"))
            if cells != n_columns:
                problems.append(
                    f"tables/{name}: row {index + 1} has {cells} cells for "
                    f"{n_columns} columns")
        for index, line in enumerate(body.split("\n"), 1):
            in_math = False
            for position, char in enumerate(line):
                if char == "$":
                    in_math = not in_math
                    continue
                if in_math or char not in "%#_":
                    continue
                if position and line[position - 1] == "\\":
                    continue
                problems.append(
                    f"tables/{name}: line {index} has an unescaped {char!r}")

    labels = set(re.findall(r"\\label\{([^}]*)\}", text)) if have_paper else set()
    refs = set(re.findall(r"\\ref\{([^}]*)\}", text)) if have_paper else set()
    for ref in sorted(refs - labels):
        problems.append(f"unresolved \\ref: {ref}")

    bib = read(os.path.join(PAPER, "references.bib")) if have_paper else ""
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite[pt]\{([^}]*)\}", text) if have_paper else []:
        cited.update(item.strip() for item in group.split(","))
    for key in sorted(cited - keys):
        problems.append(f"cited but not in bib: {key}")


    def load(name):
        path = os.path.join(OUT, name)
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    if not os.path.isdir(OUT):
        print("no outputs/ directory: skipping the numeric checks")
        print(f"sections: {len(SECTIONS)}, problems: {len(problems)}")
        for item in problems:
            print("  -", item)
        return

    checks = []
    documents = load("backbone_survey.json")["clip_openai"]
    checks.append(("documents ratio 2.12", abs(documents["ratio"] - 2.12) < 0.005))
    checks.append(("documents raw R@1 0.845", abs(documents["NN_R@1"] - 0.845) < 0.002))
    comparison = load("projection_comparison.json")
    checks.append(("closed form r=4 0.945",
                   abs(comparison["closed_form_r4"]["NN_R@1"] - 0.945) < 0.002))
    checks.append(("trained linear 0.950",
                   abs(comparison["trained_linear"]["NN_R@1"] - 0.950) < 0.002))
    checks.append(("trained per-dimension 0.876",
                   abs(comparison["trained_diagonal"]["NN_R@1"] - 0.876) < 0.002))
    large = load("domain_sae_large.json")
    checks.append(("scaled SAE dense 0.890",
                   abs(large["budgets"]["dense"]["NN_R@1"] - 0.890) < 0.002))
    checks.append(("scaled SAE k=64 0.800",
                   abs(large["budgets"]["k=64"]["NN_R@1"] - 0.800) < 0.005))
    checks.append(("scaled SAE dead latents 2.6%",
                   abs(large["dead_latent_fraction"] * 100 - 2.6) < 0.5))
    budget = load("topk_budget.json")["documents"]
    checks.append(("K budget documents active 16.8",
                   abs(budget["released_mean_active"] - 16.8) < 0.1))
    checks.append(("k=1024 0.917", abs(budget["k=1024"]["NN_R@1"] - 0.917) < 0.002))
    case = load("sae_case_study.json")["documents"]
    checks.append(("sparse code 0.112",
                   abs(case["sae:sparse_codes_clip_img"]["NN_R@1"] - 0.112) < 0.002))
    checks.append(("dense head 0.921",
                   abs(case["sae:logits_clip_img"]["NN_R@1"] - 0.921) < 0.002))
    control = load("subspace_controls.json")["control"]
    checks.append(("mismatched pairs 0.798",
                   abs(control["mismatched pairs"]["NN_R@1"] - 0.798) < 0.002))
    pair = load("pair_curve.json")
    checks.append(("K=8 0.883", abs(pair["sizes"]["8"]["NN_R@1"] - 0.883) < 0.002))
    validation = load("bound_validation.json")["layout_floor"]
    checks.append(("floor at L=6 0.694",
                   abs(validation["layouts=6"]["spread"] - 0.694) < 0.01))
    checks.append(("floor at L=48 0.230",
                   abs(validation["layouts=48"]["spread"] - 0.230) < 0.01))
    sweep = load("strength_sweep.json")
    checks.append(("sweep spread 0.313 -> 0.219",
                   abs(sweep["t0.15"]["clip_openai"]["spread"] - 0.313) < 0.005
                   and abs(sweep["t1.00"]["clip_openai"]["spread"] - 0.219) < 0.005))
    adversarial = load("adversarial_head.json")
    checks.append(("probe raw 0.802",
                   abs(adversarial["raw"]["probe_accuracy"] - 0.802) < 0.005))
    checks.append(("probe adversarial 0.583",
                   abs(adversarial["adversarial head"]["probe_accuracy"] - 0.583) < 0.005))
    prediction = load("bound_prediction.json")["summary"]
    checks.append(("proposition correlation 0.93",
                   abs(prediction["correlation"] - 0.93) < 0.02))
    checks.append(("idealised over-prediction factor 20",
                   abs(prediction["median_predicted_over_measured"] - 19.6) < 1.0))
    downstream = load("downstream_tasks.json")["clip_openai"]
    checks.append(("shapes ARI 0.080 -> 0.466",
                   abs(downstream["shapes"]["raw"]["clustering"]["ari"] - 0.080) < 0.005
                   and abs(downstream["shapes"]["rotation"]["clustering"]["ari"] - 0.466) < 0.01))
    checks.append(("pooled contents 794", downstream["pooled"]["contents"] == 794))
    identity = load("midv_full_survey.json")["clip_openai"]
    checks.append(("identity documents ratio 4.56",
                   abs(identity["ratio"] - 4.56) < 0.02))
    checks.append(("identity documents spread 0.173",
                   abs(identity["per_dim_ratio_spread"] - 0.173) < 0.005))
    checks.append(("identity documents R@1 0.790",
                   abs(identity["NN_R@1"] - 0.790) < 0.003))
    selection = load("selection_strategies.json")["documents"]
    checks.append(("per-stream top-k 0.510",
                   abs(selection["per_stream_topk"]["NN_R@1"] - 0.510) < 0.01))
    checks.append(("nucleus recovers dense 0.921",
                   abs(selection["nucleus"]["NN_R@1"] - 0.921) < 0.005))
    augmentation = load("augmentation_pairs.json")["documents/clip_openai"]
    checks.append(("augmentation pairs 0.888",
                   abs(augmentation["augmentation_pairs"]["NN_R@1"] - 0.888) < 0.005))
    stats = load("retrieval_stats.json")["documents/clip_openai"]
    checks.append(("documents effect size 4.3",
                   abs(stats["effect_size"] - 4.28) < 0.2 and stats["significant"]))
    ranking = load("rank_selection.json")["documents/clip_openai"]
    checks.append(("held-out rank 32 scores 0.957",
                   ranking["heldout_rank"] == 32
                   and abs(ranking["heldout_rank_score"] - 0.957) < 0.005
                   and ranking["heldout_rank_score"] >= ranking["fixed_r4"]))
    large = load("docvqa_scan_survey.json")["clip_openai"]
    checks.append(("large scan grid ratio 6.60",
                   abs(large["ratio"] - 6.60) < 0.02))
    checks.append(("large scan grid spread 0.211",
                   abs(large["per_dim_ratio_spread"] - 0.211) < 0.005))
    theory = load("theory_checks.json")["summary"]
    checks.append(("first-order correlation 0.998",
                   abs(theory["corr_first_order_measured"] - 0.998) < 0.002))
    checks.append(("fitted-weight attenuation 1.9x",
                   abs(theory["fitted_median_ratio"] - 1.94) < 0.05))
    fair = load("fair_baselines.json")["documents/clip_openai"]
    checks.append(("mean matching probe 0.167",
                   abs(fair["mean matching (labels)"]["probe_accuracy"] - 0.167) < 0.01))
    checks.append(("our rotation ratio 2.33",
                   abs(fair["paired-difference PCA (pairs, ours)"]["ratio"] - 2.33) < 0.02))
    points = load("backbone_survey.json")
    bands = load("heuristics_table.json")
    checks.append(("heuristics table has 7 blocks", len(bands) == 7))
    validation = load("validate_reweighting.json")["documents/clip_openai"]
    seeds = list(validation["seeds"].values())
    fitted = [s["optimised"]["clip=2"]["gain_percent"] for s in seeds]
    plugin = [s["plugin_clipped"]["gain_percent"] for s in seeds]
    checks.append(("plug-in +3.4% vs fitted +18.1% on documents",
                   abs(sum(plugin) / len(plugin) - 3.4) < 0.3
                   and abs(sum(fitted) / len(fitted) - 18.1) < 1.5))
    checks.append(("unclipped plug-in is identical",
                   abs(sum(s["plugin_unclipped"]["gain_percent"] for s in seeds) / len(seeds)
                       - sum(plugin) / len(plugin)) < 0.05))
    checks.append(("fitted costs the topic probe (0.567 -> 0.488)",
                   abs(validation["topic_probe"]["raw"] - 0.567) < 0.005
                   and abs(validation["topic_probe"]["optimised"] - 0.488) < 0.005))
    identity = load("validate_reweighting.json")["identity documents/clip_openai"]
    checks.append(("identity fitted gain +20.4%, probe 0.701 -> 0.533",
                   abs(sum(s["optimised"]["clip=2"]["gain_percent"]
                           for s in identity["seeds"].values())
                       / len(identity["seeds"]) - 20.4) < 1.0
                   and abs(identity["topic_probe"]["raw"] - 0.701) < 0.005
                   and abs(identity["topic_probe"]["optimised"] - 0.533) < 0.005))
    shapes = load("validate_reweighting.json")["shapes/siglip"]
    checks.append(("shapes/siglip fitted +57.7% and better retrieval",
                   abs(sum(s["optimised"]["clip=2"]["gain_percent"]
                           for s in shapes["seeds"].values())
                       / len(shapes["seeds"]) - 57.7) < 3.0
                   and (sum(s["optimised"]["clip=2"]["recall"]
                            for s in shapes["seeds"].values())
                        / len(shapes["seeds"])) > 0.85))
    frontier = load("diagonal_frontier.json")["documents/clip_openai"]["frontier"]
    checks.append(("unconstrained end inflates ratio, kills retrieval",
                   frontier["clip=200"]["gain_percent"] > 100
                   and frontier["clip=200"]["recall"] < 0.15))
    control = load("random_weight_control.json")["documents/clip_openai"]["clips"]["clip=2"]
    checks.append(("random weights move the ratio by ~0%",
                   abs(control["gain_mean"]) < 1.0))
    plugin_checks = load("plug_in_checks.json")
    corrs = [row["corr_fitted_vs_plug_in"] for row in plugin_checks.values()]
    checks.append(("fitted weights correlate 0.54-0.70 with plug-in weights",
                   min(corrs) > 0.50 and max(corrs) < 0.75))
    norms = [row["norm_variation_fitted"] for row in plugin_checks.values()]
    checks.append(("reweighting norm profile varies by <= 0.10",
                   max(norms) <= 0.10))
    condition = load("augmentation_condition.json")
    documents = condition["documents/clip_openai"]
    identity = condition["identity documents/clip_openai"]
    checks.append(("augmentation overlap and gain condition",
                   abs(documents["max_cosine"] - 0.66) < 0.02
                   and abs(identity["max_cosine"] - 0.86) < 0.02
                   and documents["true_basis_recall"] - documents["raw_recall"] > 0.05
                   and identity["true_basis_recall"] - identity["raw_recall"] < 0.03))

    for name, ok in checks:
        if not ok:
            problems.append(f"number mismatch: {name}")

    print(f"sections: {len(SECTIONS)}, inputs: {len(inputs)}, figures: {len(graphics)}, "
          f"labels: {len(labels)}, citations: {len(cited)}, numeric checks: {len(checks)}")
    if problems:
        print("\nPROBLEMS")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
