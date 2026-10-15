"""Publication artifacts: LaTeX tables, Markdown reports, and vector figures."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def generate_latex_table(grid_data: dict[str, Any]) -> str:
    """Generate a booktabs-compliant LaTeX table summarizing the ablation grid."""
    latex = """\\begin{table*}[t]
\\centering
\\small
\\caption{ANJANEY Ablation Grid: Screening Performance and Conformal Abstention Efficiency}
\\label{tab:ablation_grid}
\\begin{tabular}{llcccc}
\\toprule
\\textbf{Feature Modality} & \\textbf{Head} & \\textbf{RMSE [95\\% CI]} & \\textbf{AUROC [95\\% CI]} & \\textbf{AURC (Split)} & \\textbf{Status} \\\\
\\midrule
"""
    grid = grid_data["grid"]
    for mod, heads in grid.items():
        clean_mod = mod.replace("_", " ").title()
        for head, data in heads.items():
            rmse = f"{data['metrics']['rmse']['point']:.2f} [{data['metrics']['rmse']['ci_lower']:.2f}, {data['metrics']['rmse']['ci_upper']:.2f}]"
            auroc = f"{data['metrics']['auroc']['point']:.2f} [{data['metrics']['auroc']['ci_lower']:.2f}, {data['metrics']['auroc']['ci_upper']:.2f}]"
            aurc = f"{data['conformal']['split']['aurc']:.3f}"
            status = "CONFIRMATORY" if mod == "text_plus_timing" else "EXPLORATORY"
            latex += f"{clean_mod} & {head.upper()} & {rmse} & {auroc} & {aurc} & {status} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    return latex


def generate_markdown_report(
    grid_data: dict[str, Any], robustness_data: dict[str, Any], rq1_data: dict[str, Any]
) -> str:
    """Dynamically compile RESULTS.md citing exclusively real experiment JSON figures."""
    md = f"""# ANJANEY — Comprehensive Experimental Results

## 1. Confirmatory Research Questions
- **RQ1 Status:** `{rq1_data.get("status", "PENDING")}`
  - Observed Group Difference: `{rq1_data.get("statistics", {}).get("observed_group_difference", 0.0):.4f}`
  - Permutation $p$-value: `{rq1_data.get("statistics", {}).get("permutation_p_value", 1.0):.4f}`
  - Cohen's $d$: `{rq1_data.get("statistics", {}).get("cohens_d", 0.0):.3f}`
- **RQ2 (Multimodal Fusion vs Text-Only):** `{grid_data["paired_ablation"]["label"]}`
  - $\\Delta$ RMSE: `{grid_data["paired_ablation"]["delta_rmse"]:.3f}`

## 2. Modality Ablation Table
| Modality | Head | RMSE [95% CI] | AUROC [95% CI] | AURC (Split) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for mod, heads in grid_data["grid"].items():
        for head, data in heads.items():
            rmse_str = f"{data['metrics']['rmse']['point']:.2f} [{data['metrics']['rmse']['ci_lower']:.2f}, {data['metrics']['rmse']['ci_upper']:.2f}]"
            auroc_str = f"{data['metrics']['auroc']['point']:.2f} [{data['metrics']['auroc']['ci_lower']:.2f}, {data['metrics']['auroc']['ci_upper']:.2f}]"
            aurc_str = f"{data['conformal']['split']['aurc']:.3f}"
            status = "CONFIRMATORY" if mod == "text_plus_timing" else "EXPLORATORY"
            md += f"| `{mod}` | `{head}` | {rmse_str} | {auroc_str} | {aurc_str} | {status} |\n"

    md += """
## 3. Missing-Data Stress Testing
| Masking Rate | Empirical Coverage | Mean Interval Width | Abstention Rate |
| :--- | :--- | :--- | :--- |
"""
    for _, vals in robustness_data.items():
        md += f"| {vals['mask_rate'] * 100:.0f}% | {vals['empirical_coverage']:.1%} | {vals['mean_interval_width']:.2f} | {vals['abstention_rate']:.1%} |\n"

    return md


def save_publication_figures(grid_data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Modality comparison bar chart with CIs
    modalities = list(grid_data["grid"].keys())
    rmse_points = [grid_data["grid"][m]["ridge"]["metrics"]["rmse"]["point"] for m in modalities]
    y_err_low = [
        rmse_points[i] - grid_data["grid"][m]["ridge"]["metrics"]["rmse"]["ci_lower"]
        for i, m in enumerate(modalities)
    ]
    y_err_high = [
        grid_data["grid"][m]["ridge"]["metrics"]["rmse"]["ci_upper"] - rmse_points[i]
        for i, m in enumerate(modalities)
    ]

    clean_labels = [m.replace("_", " ").title() for m in modalities]

    plt.figure(figsize=(7, 4))
    plt.bar(
        clean_labels,
        rmse_points,
        yerr=[y_err_low, y_err_high],
        capsize=5,
        color="#1f77b4",
        alpha=0.85,
    )
    plt.ylabel("PHQ-8 RMSE (95% CI)")
    plt.title("Modality Ablation: Prediction Error Across Feature Sets")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "modality_ablation.png", dpi=300)
    plt.savefig(output_dir / "modality_ablation.svg")
    plt.close()
