
import json
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

from notebooks.utils.dtos import Acquisition, Merger, Other


def get_gt_pydantic_model(gold_data: dict[str, Any]) -> Merger | Acquisition | Other:
    article_type : str = gold_data["article_type"]

    if article_type == "merger":
        return Merger(**gold_data)
    elif article_type == "acquisition":
        return Acquisition(**gold_data)
    else:  # article_type == "other"
        return Other(**gold_data)

def extract_first_n_sentences(text: str, num_sentences: int = 3) -> str:
    # Define sentence boundary pattern: period followed by space or newline
    pattern = r"\.(?:\s+|\n+)"

    # Split the text into sentences
    sentences = re.split(pattern, text)

    # Filter out empty sentences and join the first 3 with periods
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    first_n = valid_sentences[:num_sentences]

    # Join with periods and spaces
    result = ". ".join(first_n) + "."
    return result

def read_data(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)
    
def visualize_model_scores(
    models_configs: dict[str, Any], 
    save_path: Path | str | None = None
) -> None:
    """
    Creates a side-by-side visualization of train and test scores for each model.
    
    Args:
        models_configs: Dictionary mapping model names to their train/test scores.
        save_path: Optional path to save the figure as SVG. If None, figure is not saved.
    """
    # Extract data
    model_names = list(models_configs.keys())
    train_scores = [models_configs[m]["train_score"] for m in model_names]
    test_scores = [models_configs[m]["test_score"] for m in model_names]

    model_names = [m.upper() for m in model_names]

    # Set up the style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train scores
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, train_scores, color=palette, edgecolor="black", linewidth=1.2)
    ax1.set_title("Train Scores", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=16)
    ax1.set_ylim(0, 100)
    ax1.bar_label(bars1, fmt="%.2f", fontsize=12, padding=3)
    
    # Test scores1
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, test_scores, color=palette, edgecolor="black", linewidth=1.2)
    ax2.set_title("Test Scores", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Score (%)", fontsize=16)
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt="%.2f", fontsize=12, padding=3)
    
    # Styling
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
    
    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Save figure as SVG if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".png")
        fig.savefig(save_path, format="png", bbox_inches="tight")
    
    plt.show()


def combine_programs(gepa_path: str, fewshot_path: str, output_path: str) -> None:
    """
    Combines GEPA-optimized instructions with FewShot demos.
    
    Args:
        gepa_path: Path to the GEPA program JSON file
        fewshot_path: Path to the FewShot program JSON file
        output_path: Path to save the combined program
    """
    # Load both program files
    with open(gepa_path, "r") as f:
        gepa_program_dict = json.load(f)

    with open(fewshot_path, "r") as f:
        fewshot_program_dict = json.load(f)

    # Add demos from fewshot_extractor to gepa_program for each component
    for component in ["classifier", "merger_extractor", "acquisition_extractor"]:
        if component in fewshot_program_dict and component in gepa_program_dict:
            gepa_program_dict[component]["demos"] = fewshot_program_dict[component]["demos"]

    # Save the combined program
    with open(output_path, "w") as f:
        json.dump(gepa_program_dict, f, indent=2)

    # Verify the demos were added
    for component in ["classifier", "merger_extractor", "acquisition_extractor"]:
        demo_count = len(gepa_program_dict[component]["demos"])
        print(f"{component}: {demo_count} demos added")

    print(f"\nCombined program saved to: {output_path}")