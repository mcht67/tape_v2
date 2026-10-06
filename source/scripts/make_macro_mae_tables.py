#!/usr/bin/env python3
"""
Build LaTeX tables for every macro-MAE(1-6) result already computed under
plots/figures/, combined into one file per experiment:

    results/result_tables_macro_mae_pooled.tex          (Pooled-Embeddings; RQ1 + RQ5 difficulty)
    results/result_tables_macro_mae_multi_task.tex      (Spatial-Embeddings TC-head configs; RQ2)
    results/result_tables_macro_mae_XCM_generalization.tex (region-specific vs. XCM-frozen vs. XCM-fine-tuned; RQ5)

Reads the JSON / markdown outputs of compute_rq1_macro_mae_table.py,
compute_rq2_macro_mae_paired_analysis.py, compute_rq5_macro_mae_summary.py and
compute_rq5_macro_mae_difficulty_robustness.py (re-run those first to refresh),
so the tables cannot drift from the reported numbers. Backbone names come from
backbone_meta.py; best (and tied-best) values are bold; lower is better.
There is no macro-MAE result for the Fine-Tuning experiment (only the XCM
fine-tuned models, in the XCM file). Correlation/regression analyses built on
macro-MAE (rq5 difficulty robustness) are not tabulated here, only its
per-dataset macro-MAE values.

Usage:
    complete-venv/bin/python source/scripts/make_macro_mae_tables.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_result_tables as mrt
from backbone_meta import BACKBONE_META, ordered_backbones

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG = REPO_ROOT / "plots" / "figures"
RESULTS_DIR = REPO_ROOT / "results"

DATASETS = ["HSN", "NES", "PER", "POW", "SNE", "SSW", "UHH"]
ORDER = ordered_backbones()
PM = r"$\pm$"


def name(canon):
    return mrt.escape_latex(BACKBONE_META[canon]["display"])


def f2(x):
    return f"{x:.2f}"


def ms(mean, sd):
    m = f"{mean:.2f}"
    return f"{'0.00' if m == '-0.00' else m} {PM} {sd:.2f}"


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def bold_best(cells_vals, lower=True):
    """cells_vals: list of (text, value) -> list of text with best/tied-best bolded."""
    vals = [v for _, v in cells_vals]
    best = min(vals) if lower else max(vals)
    return [mrt.bold(t) if v == best else t for t, v in cells_vals]


def table(caption, label, colspec, header, body_lines, wide=False):
    env = "table*" if wide else "table"
    out = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if wide:
        out.append(r"\resizebox{\textwidth}{!}{%")
    out += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule"] + body_lines
    out += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        out.append("}")
    out.append(rf"\end{{{env}}}")
    return "\n".join(out)


def document(sections):
    parts = [mrt.DOC_PREAMBLE]
    for title, tables in sections:
        parts.append(rf"\section*{{{title}}}" + "\n")
        parts += [t + "\n" for t in tables]
    parts.append(mrt.DOC_END)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pooled-Embeddings (RQ1 + per-dataset difficulty from RQ5)
# ---------------------------------------------------------------------------

def pooled_tables():
    rq1 = json.load(open(FIG / "rq1" / "rq1_macro_mae_table.json"))
    diff = json.load(open(FIG / "rq5" / "rq5_macro_mae_difficulty_robustness.json"))
    backbones = [b for b in ORDER if b in rq1]
    tables = []

    syn = [(f"{rq1[b]['synthetic']['summary']['mean']:.2f} {PM} {rq1[b]['synthetic']['summary']['sd']:.2f}",
            rq1[b]["synthetic"]["summary"]["mean"]) for b in backbones]
    sc = [(f"{rq1[b]['soundscape']['summary']['mean']:.2f} {PM} {rq1[b]['soundscape']['summary']['sd']:.2f}",
           rq1[b]["soundscape"]["summary"]["mean"]) for b in backbones]
    syn_c, sc_c = bold_best(syn), bold_best(sc)
    body = [f"{name(b)} & {syn_c[i]} & {sc_c[i]} \\\\" for i, b in enumerate(backbones)]
    tables.append(table(
        r"Macro-MAE(1-6) $\downarrow$ on synthetic (test) and soundscape (test) data, mean ($\pm$ std) across the 7 region-matched "
        r"datasets, per backbone (regression head, frozen backbone with pooled-embedding MLP). Macro-MAE(1-6) is the unweighted mean "
        r"of per-level MAE over true polyphony levels 1--6, computed per dataset first. Soundscape uses unambiguous-label clips only.",
        "tab:macro_mae_pooled_summary", "lcc", r"Backbone & Synthetic & Soundscape \\", body))

    for split, key, label in [("synthetic", "synthetic", "tab:macro_mae_pooled_per_region_synthetic"),
                              ("soundscape", "soundscape", "tab:macro_mae_pooled_per_region_soundscape")]:
        body = []
        for b in backbones:
            cells = []
            for r in DATASETS:
                cell = rq1[b][key]["per_region"][r]
                txt = f2(cell["macro_mae_1_6"])
                if cell["n_levels_used"] < 6:
                    txt += rf"\textsuperscript{{{cell['n_levels_used']}}}"
                cells.append(txt)
            body.append(f"{name(b)} & " + " & ".join(cells) + r" \\")
        note = (r" Superscripts give the number of levels (of 6) with data that the dataset's average was taken over, where fewer than 6."
                if key == "soundscape" else "")
        tables.append(table(
            rf"Macro-MAE(1-6) $\downarrow$ on {split} test data, per backbone and dataset (regression head).{note}",
            label, "l" + "c" * len(DATASETS), "Backbone & " + " & ".join(DATASETS) + r" \\", body, wide=True))

    cov = diff["level_coverage"]
    m6, sd6, plain = diff["macro_mae_by_dataset"], diff["macro_mae_sd_by_dataset"], diff["plain_mae_by_dataset"]
    m123, sd123 = diff["common_mae_by_dataset"], diff["common_mae_sd_by_dataset"]
    c1 = bold_best([(ms(m6[d], sd6[d]), m6[d]) for d in DATASETS])
    c2 = bold_best([(ms(m123[d], sd123[d]), m123[d]) for d in DATASETS])
    c3 = bold_best([(f2(plain[d]), plain[d]) for d in DATASETS])
    body = [f"{d} & {c1[i]} & {c2[i]} & {c3[i]} & {cov[d]['n_levels_with_data']} \\\\" for i, d in enumerate(DATASETS)]
    tables.append(table(
        r"Soundscape difficulty per dataset, averaged (mean $\pm$ std) across the 13 pooled-embedding backbones: macro-MAE over levels 1--6, "
        r"macro-MAE over the levels 1--3 present in every dataset (equal level coverage), and plain (pooled) range MAE. "
        r"The last column gives the number of levels 1--6 with at least one unambiguous clip.",
        "tab:macro_mae_pooled_per_dataset_difficulty", "lcccc",
        r"Dataset & Macro-MAE(1-6) $\downarrow$ & Macro-MAE(1-3) $\downarrow$ & MAE $\downarrow$ & Levels with data \\", body))
    return [("Pooled-Embeddings: macro-MAE results", tables)]


# ---------------------------------------------------------------------------
# Spatial-Embeddings / multi-task (RQ2)
# ---------------------------------------------------------------------------

CONFIG_SHORT = {
    "pooled-MLP (no TC-head)": "Pooled-MLP",
    "+TC-head": "TC",
    "+TC-head+polyphony classification": "TC + polyphony classification",
    "+TC-head+frame-level polyphony": "TC + framewise polyphony",
    "+TC-head+frame-level call activity": "TC + event logits",
    "+TC-head+both-aux": "TC + event logits, framewise polyphony",
}
DISPLAY_TO_CANON = {v["display"]: k for k, v in BACKBONE_META.items()}


def parse_plain_rq2(md_text, backbone_order):
    """{(canonical_backbone, config_label): (mean, sd)} from the '## 0. Plain macro-MAE' table.
    Rows are keyed by order of first appearance (matching backbone_order) rather than by the
    markdown's display string, which can go stale relative to backbone_meta.py."""
    sec = md_text.split("## 0.")[1].split("## 1.")[0]
    out, seen = {}, []
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and "±" in cells[2]:
            if cells[0] not in seen:
                seen.append(cells[0])
            mean, sd = (float(x) for x in cells[2].split("±"))
            out[(backbone_order[seen.index(cells[0])], cells[1])] = (mean, sd)
    return out


def paired_body(pooled, per_backbone, backbone_order, configs):
    pooled_body = []
    for c in configs:
        s = pooled[c]
        pooled_body.append(f"{CONFIG_SHORT[c]} & {ms(s['mean'], s['std'])} & {pfmt(s['t_p'])} & {pfmt(s['wilcoxon_p'])} \\\\")
    per_body = []
    for b in backbone_order:
        for i, c in enumerate(configs):
            s = per_backbone[c][b]
            first = rf"\multirow{{{len(configs)}}}{{*}}{{{name(b)}}}" if i == 0 else ""
            per_body.append(f"{first} & {CONFIG_SHORT[c]} & {ms(s['mean'], s['std'])} & {pfmt(s['t_p'])} & {pfmt(s['wilcoxon_p'])} \\\\")
        per_body.append(r"\cmidrule(lr){1-2}")
    per_body.pop()
    return pooled_body, per_body


def multitask_tables():
    md = (FIG / "rq2" / "rq2_macro_mae_paired_analysis.md").read_text()
    js = json.load(open(FIG / "rq2" / "rq2_macro_mae_paired_analysis.json"))
    backbone_order = js["backbone_order"]
    plain = parse_plain_rq2(md, backbone_order)
    configs_all = list(CONFIG_SHORT)
    aux_configs = [c for c in configs_all if c not in ("pooled-MLP (no TC-head)", "+TC-head")]
    tables = []

    body = []
    for b in backbone_order:
        cells = bold_best([(ms(*plain[(b, c)]), plain[(b, c)][0]) for c in configs_all])
        for i, c in enumerate(configs_all):
            first = rf"\multirow{{{len(configs_all)}}}{{*}}{{{name(b)}}}" if i == 0 else ""
            body.append(f"{first} & {CONFIG_SHORT[c]} & {cells[i]} \\\\")
        body.append(r"\cmidrule(lr){1-2}")
    body.pop()
    tables.append(table(
        r"Macro-MAE(1-6) $\downarrow$ on soundscape test data, mean ($\pm$ std) across the 7 datasets, per backbone and head configuration "
        r"(Pooled-MLP = frozen-backbone pooled-embedding MLP; TC = temporal-convolution head on spatial embeddings; + = auxiliary objectives). "
        r"Best per backbone in bold.",
        "tab:macro_mae_multitask_by_backbone", "llc", r"Backbone & Head & Macro-MAE(1-6) $\downarrow$ \\", body))

    cfg_stats = []
    import statistics
    for c in configs_all:
        means = [plain[(b, c)][0] for b in backbone_order]
        cfg_stats.append((statistics.mean(means), statistics.stdev(means)))
    cells = bold_best([(ms(m, s), m) for m, s in cfg_stats])
    body = [f"{CONFIG_SHORT[c]} & {cells[i]} \\\\" for i, c in enumerate(configs_all)]
    tables.append(table(
        r"Macro-MAE(1-6) $\downarrow$ on soundscape test data, averaged (mean $\pm$ std) across the 5 backbones (each backbone's mean over 7 datasets), "
        r"for each head configuration.",
        "tab:macro_mae_multitask_by_config", "lc", r"Head & Macro-MAE(1-6) $\downarrow$ \\", body))

    body = []
    for b in backbone_order:
        best = js["best_aux"][b]
        base = plain[(b, "pooled-MLP (no TC-head)")][0]
        body.append(f"{name(b)} & {CONFIG_SHORT[best['best_config']]} & {best['best_mean_macro_mae']:.2f} & {base:.2f} \\\\")
    tables.append(table(
        r"Best auxiliary configuration per backbone by soundscape macro-MAE(1-6) $\downarrow$, against the pooled-MLP baseline.",
        "tab:macro_mae_multitask_best_aux", "llcc", r"Backbone & Best auxiliary configuration & Macro-MAE(1-6) & Pooled-MLP \\", body))

    for ref, pkey, bkey, label, refdesc in [
        ("pooled-MLP", "pooled_vs_baseline", "per_backbone_vs_baseline", "vs_pooled_mlp", "the pooled-MLP baseline"),
        ("TC alone", "pooled_vs_tc_alone", "per_backbone_vs_tc_alone", "vs_tc_alone", "the TC head without auxiliary objectives"),
    ]:
        pooled_body, per_body = paired_body(js[pkey], js[bkey], backbone_order, aux_configs)
        tables.append(table(
            rf"Paired difference in soundscape macro-MAE(1-6) of each TC + auxiliary configuration against {refdesc} "
            rf"(negative = improvement), pooled over 5 backbones $\times$ 7 datasets ($n=35$); mean $\pm$ std of the difference, "
            rf"paired $t$-test and Wilcoxon signed-rank $p$-values.",
            f"tab:macro_mae_multitask_pooled_{label}", "lccc", r"Head & $\Delta$ macro-MAE(1-6) & $t$-test $p$ & Wilcoxon $p$ \\", pooled_body))
        tables.append(table(
            rf"As Table~\ref{{tab:macro_mae_multitask_pooled_{label}}}, per backbone ($n=7$ datasets each).",
            f"tab:macro_mae_multitask_per_backbone_{label}", "llccc",
            r"Backbone & Head & $\Delta$ macro-MAE(1-6) & $t$-test $p$ & Wilcoxon $p$ \\", per_body))
    return [("Spatial-Embeddings (multi-task): macro-MAE results", tables)]


# ---------------------------------------------------------------------------
# XCM generalization (RQ5)
# ---------------------------------------------------------------------------

GROUPS = [("region_specific", "Region-specific"), ("xcm_frozen", "XCM-frozen"), ("xcm_finetuned", "XCM-fine-tuned")]
XCM_DATASETS = ["HSN", "NES", "PER", "POW", "SNE", "SSW", "UHH"]


def xcm_tables():
    js = json.load(open(FIG / "rq5" / "rq5_macro_mae_summary.json"))
    backbones = [b for b in ORDER if b in js["backbone_order"]]
    tables = []

    pm_ = js["pooled_macro_mae"]
    rm_ = js["pooled_range_mae_reference"]
    c1 = bold_best([(ms(pm_[k]["mean"], pm_[k]["std"]), pm_[k]["mean"]) for k, _ in GROUPS])
    c2 = bold_best([(ms(rm_[k]["mean"], rm_[k]["std"]), rm_[k]["mean"]) for k, _ in GROUPS])
    body = [f"{lab} & {c1[i]} & {c2[i]} \\\\" for i, (_, lab) in enumerate(GROUPS)]
    tables.append(table(
        r"Soundscape test results averaged (mean $\pm$ std) over all 28 cells (4 backbones $\times$ 7 datasets), for each model grouping: "
        r"macro-MAE(1-6) and plain range MAE (lower is better).",
        "tab:macro_mae_xcm_overall", "lcc", r"Model grouping & Macro-MAE(1-6) $\downarrow$ & MAE $\downarrow$ \\", body))

    pdm = js["per_dataset"]
    body = []
    for d in XCM_DATASETS:
        cells = bold_best([(ms(pdm[k][d]["mean"], pdm[k][d]["std"]), pdm[k][d]["mean"]) for k, _ in GROUPS])
        body.append(f"{d} & " + " & ".join(cells) + r" \\")
    tables.append(table(
        r"Macro-MAE(1-6) $\downarrow$ on soundscape test data per dataset, mean $\pm$ std across the 4 backbones, for each model grouping. Best per dataset in bold.",
        "tab:macro_mae_xcm_per_dataset", "lccc", "Dataset & " + " & ".join(l for _, l in GROUPS) + r" \\", body))

    pbb = js["per_backbone"]
    body = []
    for b in backbones:
        cells = bold_best([(ms(pbb[k][b]["mean"], pbb[k][b]["std"]), pbb[k][b]["mean"]) for k, _ in GROUPS])
        body.append(f"{name(b)} & " + " & ".join(cells) + r" \\")
    tables.append(table(
        r"Macro-MAE(1-6) $\downarrow$ on soundscape test data per backbone, mean $\pm$ std across the 7 datasets, for each model grouping. Best per backbone in bold.",
        "tab:macro_mae_xcm_per_backbone", "lccc", "Backbone & " + " & ".join(l for _, l in GROUPS) + r" \\", body))

    body = []
    for comp, s in js["cell_level_deltas"].items():
        st = s["sign_test"]
        comp_tex = mrt.escape_latex(comp.replace("XCM frozen", "XCM-frozen").replace("XCM fine-tuned", "XCM-fine-tuned"))
        body.append(f"{comp_tex} & {ms(s['mean'], s['std'])} & {pfmt(s['t_p'])} & {pfmt(s['wilcoxon_p'])} & "
                    f"{st['wins']}--{st['losses']}--{st['ties']} & {pfmt(st['p_two_sided'])} \\\\")
    tables.append(table(
        r"Paired cell-level comparisons of macro-MAE(1-6) on soundscape test data (28 backbone $\times$ dataset cells; difference = first minus second, "
        r"negative = first is better): mean $\pm$ std, paired $t$-test and Wilcoxon $p$-values, and sign test (wins--losses--ties, two-sided $p$).",
        "tab:macro_mae_xcm_paired", "lccccc",
        r"Comparison & $\Delta$ macro-MAE(1-6) & $t$-test $p$ & Wilcoxon $p$ & W--L--T & Sign test $p$ \\", body, wide=True))

    for k, lab in GROUPS:
        body = []
        for b in backbones:
            cells = [f2(js["per_cell"][k][b][d]) if js["per_cell"][k][b].get(d) is not None else "--" for d in XCM_DATASETS]
            body.append(f"{name(b)} & " + " & ".join(cells) + r" \\")
        tables.append(table(
            rf"Macro-MAE(1-6) $\downarrow$ on soundscape test data per backbone and dataset, {lab}.",
            f"tab:macro_mae_xcm_cells_{k}", "l" + "c" * len(XCM_DATASETS), "Backbone & " + " & ".join(XCM_DATASETS) + r" \\", body))
    return [("XCM generalization: macro-MAE results", tables)]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    for fname, sections in [
        ("result_tables_macro_mae_pooled.tex", pooled_tables()),
        ("result_tables_macro_mae_multi_task.tex", multitask_tables()),
        ("result_tables_macro_mae_XCM_generalization.tex", xcm_tables()),
    ]:
        (RESULTS_DIR / fname).write_text(document(sections))
        print(f"Wrote {RESULTS_DIR / fname}")


if __name__ == "__main__":
    main()
