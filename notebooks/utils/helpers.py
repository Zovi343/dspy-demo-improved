
import json
import re
from pathlib import Path
from typing import Any

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