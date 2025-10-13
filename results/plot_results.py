"""Visualize evaluation accuracy per article broken down by field for each run."""
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


ARTICLE_IDS = [17, 15, 18, 19, 16, 20, 22, 21, 28, 23]
FIELDS = [
    "child_company",
    "child_company_ticker",
    "company_1",
    "company_1_ticker",
    "company_2",
    "company_2_ticker",
    "deal_amount",
    "deal_currency",
    "merged_entity",
    "parent_company",
    "parent_company_ticker",
]
RUN_FIELD_MISMATCHES = {
    "Baseline": {
        "child_company": {15},
        "child_company_ticker": set(),
        "company_1": set(),
        "company_1_ticker": {18},
        "company_2": set(),
        "company_2_ticker": {18},
        "deal_amount": {15, 17, 18, 19, 20, 22, 28},
        "deal_currency": {15},
        "merged_entity": set(),
        "parent_company": {15},
        "parent_company_ticker": set(),
    },
    "Bootstrap Fewshot": {
        "child_company": set(),
        "child_company_ticker": set(),
        "company_1": set(),
        "company_1_ticker": {18},
        "company_2": set(),
        "company_2_ticker": {18},
        "deal_amount": {17, 18, 19, 20, 22, 28},
        "deal_currency": set(),
        "merged_entity": set(),
        "parent_company": set(),
        "parent_company_ticker": set(),
    },
    "GEPA": {
        "child_company": set(),
        "child_company_ticker": set(),
        "company_1": set(),
        "company_1_ticker": {18},
        "company_2": set(),
        "company_2_ticker": {18},
        "deal_amount": set(),
        "deal_currency": set(),
        "merged_entity": set(),
        "parent_company": set(),
        "parent_company_ticker": set(),
    },
}


def build_accuracy_matrix(article_ids, fields, mismatches_by_field):
    """Return field-by-article matrix with 1 for correct predictions, 0 for mismatches."""
    rows = []
    for field in fields:
        mismatches = mismatches_by_field.get(field, set())
        rows.append([0 if article_id in mismatches else 1 for article_id in article_ids])
    return np.array(rows)


def plot_heatmaps(article_ids, fields, run_field_mismatches):
    """Plot pastel red/green heatmaps for each evaluation run."""
    run_names = list(run_field_mismatches.keys())
    fig_height = max(6, len(run_names) * len(fields) * 0.22)
    fig, axes = plt.subplots(
        len(run_names),
        1,
        sharex=True,
        figsize=(len(article_ids) * 0.7, fig_height),
    )
    if len(run_names) == 1:
        axes = [axes]

    cmap = ListedColormap(["#f9c4c4", "#c8f7c5"])  # pastel red, pastel green
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    for ax, run_name in zip(axes, run_names):
        data = build_accuracy_matrix(article_ids, fields, run_field_mismatches[run_name])
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields, fontsize=8)
        ax.set_title(run_name, fontsize=10, fontweight="bold")

        for (row_idx, col_idx), value in np.ndenumerate(data):
            label = "✔️" if value else "X"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="#1f1f1f",
                fontsize=8,
            )

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], pad=0.01, shrink=0.9)
        cbar.ax.set_yticklabels(["Mismatch", "Correct"])

    axes[-1].set_xticks(range(len(article_ids)))
    axes[-1].set_xticklabels(article_ids)
    axes[-1].set_xlabel("Article ID")

    # fig.suptitle("Evaluation accuracy (correct/incorrect) per article", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    """Generate stacked heatmaps and display them."""
    fig = plot_heatmaps(ARTICLE_IDS, FIELDS, RUN_FIELD_MISMATCHES)
    plt.show()
    return fig


if __name__ == "__main__":
    main()
