"""Evaluation benchmark dataset loader and data model."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalSample:
    """A benchmark Q&A evaluation sample."""

    sample_id: str
    query: str
    ground_truth: str
    expected_route: str
    relevant_chunks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_eval_dataset(file_path: str | Path) -> list[EvalSample]:
    """Load JSONL benchmark dataset for RAG and agent evaluation."""
    samples: list[EvalSample] = []
    path = Path(file_path)
    if not path.exists():
        return samples

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            samples.append(
                EvalSample(
                    sample_id=data.get("sample_id", ""),
                    query=data.get("query", ""),
                    ground_truth=data.get("ground_truth", ""),
                    expected_route=data.get("expected_route", "document_search"),
                    relevant_chunks=data.get("relevant_chunks", []),
                    metadata=data.get("metadata", {}),
                )
            )

    return samples
