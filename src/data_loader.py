"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """A single data sample with input and expected output."""

    id: str
    input_text: str
    expected_output: str
    metadata: dict[str, Any] | None = None


class DataLoader:
    """Loads and preprocesses evaluation datasets.

    Supported formats: JSON, JSONL, CSV.
    """

    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv"}

    def __init__(self, data_dir: str) -> None:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir

    def load(self, filename: str) -> list[Sample]:
        """Load samples from a file."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Use: {self.SUPPORTED_EXTENSIONS}"
            )

        if ext == ".json":
            return self._load_json(path)
        if ext == ".jsonl":
            return self._load_jsonl(path)
        return self._load_csv(path)

    def load_all(self) -> list[Sample]:
        """Load all supported data files from the directory."""
        samples: list[Sample] = []
        for fname in sorted(os.listdir(self.data_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                samples.extend(self.load(fname))
        return samples

    @staticmethod
    def _load_json(path: str) -> list[Sample]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("samples", data.get("data", [data]))
        return [DataLoader._dict_to_sample(d) for d in data]

    @staticmethod
    def _load_jsonl(path: str) -> list[Sample]:
        samples: list[Sample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(DataLoader._dict_to_sample(json.loads(line)))
        return samples

    @staticmethod
    def _load_csv(path: str) -> list[Sample]:
        samples: list[Sample] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                samples.append(
                    Sample(
                        id=row.get("id", str(i)),
                        input_text=row.get("input", row.get("question", "")),
                        expected_output=row.get("output", row.get("answer", "")),
                        metadata={k: v for k, v in row.items() if k not in {"id", "input", "output", "question", "answer"}},
                    )
                )
        return samples

    @staticmethod
    def _dict_to_sample(d: dict[str, Any]) -> Sample:
        return Sample(
            id=str(d.get("id", "")),
            input_text=str(d.get("input", d.get("question", ""))),
            expected_output=str(d.get("output", d.get("answer", ""))),
            metadata=d.get("metadata"),
        )

    @staticmethod
    def save_samples(samples: list[Sample], path: str) -> None:
        """Save samples to a JSON file."""
        data = [
            {
                "id": s.id,
                "input": s.input_text,
                "output": s.expected_output,
                "metadata": s.metadata,
            }
            for s in samples
        ]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
