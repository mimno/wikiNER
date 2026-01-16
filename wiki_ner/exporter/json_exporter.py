"""Export annotated data to JSON format."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class JSONExporter:
    """Export NER training data to JSON format."""

    def __init__(self, pretty: bool = True):
        """Initialize exporter.

        Args:
            pretty: If True, output indented JSON.
        """
        self.pretty = pretty

    def export(
        self,
        data: List[Dict],
        output_path: str,
        append: bool = False,
    ) -> int:
        """Export data to a JSON file.

        Args:
            data: List of annotation dicts with "text" and "entities".
            output_path: Path to output file.
            append: If True, append to existing file.

        Returns:
            Number of examples written.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_data = []
        if append and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

        combined = existing_data + data

        with open(path, "w", encoding="utf-8") as f:
            if self.pretty:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            else:
                json.dump(combined, f, ensure_ascii=False)

        return len(data)

    def export_single(
        self, text: str, entities: List[List], output_path: str
    ) -> None:
        """Export a single annotated example.

        Args:
            text: The text content.
            entities: List of [start, end, label] entity spans.
            output_path: Path to output file.
        """
        self.export([{"text": text, "entities": entities}], output_path)

    @staticmethod
    def load(input_path: str) -> List[Dict]:
        """Load data from a JSON file.

        Args:
            input_path: Path to JSON file.

        Returns:
            List of annotation dicts.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate(data: List[Dict]) -> List[str]:
        """Validate JSON training data format.

        Args:
            data: List of annotation dicts.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item {i}: Expected dict, got {type(item)}")
                continue

            if "text" not in item:
                errors.append(f"Item {i}: Missing 'text' field")

            if "entities" not in item:
                errors.append(f"Item {i}: Missing 'entities' field")
                continue

            text = item.get("text", "")
            for j, ent in enumerate(item.get("entities", [])):
                if not isinstance(ent, (list, tuple)) or len(ent) != 3:
                    errors.append(
                        f"Item {i}, entity {j}: Expected [start, end, label]"
                    )
                    continue

                start, end, label = ent
                if not isinstance(start, int) or not isinstance(end, int):
                    errors.append(
                        f"Item {i}, entity {j}: start/end must be integers"
                    )
                    continue

                if start < 0 or end > len(text) or start >= end:
                    errors.append(
                        f"Item {i}, entity {j}: Invalid span [{start}, {end}] "
                        f"for text of length {len(text)}"
                    )

        return errors


def merge_json_files(
    input_paths: List[str], output_path: str, dedupe: bool = True
) -> int:
    """Merge multiple JSON files into one.

    Args:
        input_paths: List of input file paths.
        output_path: Output file path.
        dedupe: If True, remove duplicate texts.

    Returns:
        Total number of examples in merged file.
    """
    all_data = []
    seen_texts = set()

    for path in input_paths:
        data = JSONExporter.load(path)
        for item in data:
            if dedupe:
                text_key = item.get("text", "")[:100]  # Use first 100 chars as key
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
            all_data.append(item)

    exporter = JSONExporter()
    exporter.export(all_data, output_path)
    return len(all_data)
