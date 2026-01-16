"""Export annotated data to spaCy training format.

This module converts JSON NER annotations to spaCy's DocBin format
for training custom NER models.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example


class SpacyExporter:
    """Export NER training data to spaCy format."""

    def __init__(self, lang: str = "en"):
        """Initialize the exporter.

        Args:
            lang: Language code for spaCy blank model.
        """
        self.lang = lang
        self.nlp = spacy.blank(lang)

    def convert_to_docbin(
        self,
        data: List[Dict],
        output_path: str,
        validate: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """Convert JSON annotations to spaCy DocBin format.

        Args:
            data: List of dicts with "text" and "entities" keys.
            output_path: Path for output .spacy file.
            validate: If True, validate entity spans before adding.

        Returns:
            Tuple of (successful_docs, skipped_docs, error_messages).
        """
        db = DocBin()
        successful = 0
        skipped = 0
        errors = []

        for i, item in enumerate(data):
            text = item.get("text", "")
            entities = item.get("entities", [])

            if not text:
                skipped += 1
                continue

            try:
                doc = self.nlp.make_doc(text)
                ents = []

                for ent_data in entities:
                    start, end, label = ent_data[0], ent_data[1], ent_data[2]

                    # Validate span
                    if validate:
                        error = self._validate_span(text, start, end, label)
                        if error:
                            errors.append(f"Item {i}: {error}")
                            continue

                    # Create span with alignment
                    span = doc.char_span(
                        start, end, label=label, alignment_mode="contract"
                    )

                    if span is None:
                        # Try expanding alignment
                        span = doc.char_span(
                            start, end, label=label, alignment_mode="expand"
                        )

                    if span is not None:
                        ents.append(span)
                    else:
                        errors.append(
                            f"Item {i}: Could not align span [{start}:{end}] '{text[start:end][:30]}'"
                        )

                # Filter overlapping entities (keep longer ones)
                ents = self._filter_overlapping(ents)

                doc.ents = ents
                db.add(doc)
                successful += 1

            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")
                skipped += 1

        # Save DocBin
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        db.to_disk(output_path)

        return successful, skipped, errors

    def _validate_span(
        self, text: str, start: int, end: int, label: str
    ) -> Optional[str]:
        """Validate an entity span.

        Args:
            text: Full text.
            start: Start character offset.
            end: End character offset.
            label: Entity label.

        Returns:
            Error message if invalid, None if valid.
        """
        if start < 0:
            return f"Negative start offset: {start}"

        if end > len(text):
            return f"End offset {end} exceeds text length {len(text)}"

        if start >= end:
            return f"Invalid span: start ({start}) >= end ({end})"

        if label not in {"PER", "LOC", "ORG", "MISC"}:
            return f"Unknown label: {label}"

        return None

    def _filter_overlapping(self, spans: List) -> List:
        """Filter overlapping spans, keeping longer ones.

        Args:
            spans: List of spaCy Span objects.

        Returns:
            Filtered list with no overlaps.
        """
        if not spans:
            return []

        # Sort by start position, then by length (descending)
        sorted_spans = sorted(spans, key=lambda s: (s.start_char, -(s.end_char - s.start_char)))

        filtered = []
        last_end = -1

        for span in sorted_spans:
            if span.start_char >= last_end:
                filtered.append(span)
                last_end = span.end_char

        return filtered

    def split_data(
        self,
        data: List[Dict],
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/dev/test sets.

        Args:
            data: List of annotation dicts.
            train_ratio: Proportion for training (default 0.8).
            dev_ratio: Proportion for development (default 0.1).
            test_ratio: Proportion for testing (default 0.1).
            shuffle: If True, shuffle before splitting.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_data, dev_data, test_data).
        """
        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")

        if shuffle:
            random.seed(seed)
            data = data.copy()
            random.shuffle(data)

        n = len(data)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)

        train_data = data[:train_end]
        dev_data = data[train_end:dev_end]
        test_data = data[dev_end:]

        return train_data, dev_data, test_data

    def convert_and_split(
        self,
        input_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[str, Tuple[int, int]]:
        """Load JSON, split, and convert to spaCy format.

        Args:
            input_path: Path to JSON file with annotations.
            output_dir: Directory for output .spacy files.
            train_ratio: Proportion for training.
            dev_ratio: Proportion for development.
            test_ratio: Proportion for testing.
            shuffle: If True, shuffle before splitting.
            seed: Random seed.

        Returns:
            Dict with counts: {"train": (success, skip), "dev": ..., "test": ...}
        """
        # Load JSON data
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Split data
        train_data, dev_data, test_data = self.split_data(
            data, train_ratio, dev_ratio, test_ratio, shuffle, seed
        )

        output_dir = Path(output_dir)
        results = {}

        # Convert each split
        for name, split_data in [
            ("train", train_data),
            ("dev", dev_data),
            ("test", test_data),
        ]:
            if split_data:
                output_path = output_dir / f"{name}.spacy"
                success, skip, errors = self.convert_to_docbin(
                    split_data, str(output_path)
                )
                results[name] = (success, skip)

                if errors:
                    # Write errors to log file
                    error_path = output_dir / f"{name}_errors.txt"
                    with open(error_path, "w") as f:
                        f.write("\n".join(errors))

        return results

    def get_label_stats(self, data: List[Dict]) -> Dict[str, int]:
        """Get entity label statistics from data.

        Args:
            data: List of annotation dicts.

        Returns:
            Dict mapping labels to counts.
        """
        stats = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0, "total": 0}

        for item in data:
            for ent in item.get("entities", []):
                label = ent[2] if len(ent) > 2 else "MISC"
                if label in stats:
                    stats[label] += 1
                stats["total"] += 1

        return stats

    def generate_config(
        self,
        output_path: str,
        train_path: str,
        dev_path: str,
        vectors: Optional[str] = None,
    ) -> str:
        """Generate a basic spaCy training config.

        Args:
            output_path: Path to save config file.
            train_path: Path to train.spacy file.
            dev_path: Path to dev.spacy file.
            vectors: Optional path to word vectors.

        Returns:
            Config file content as string.
        """
        config = f'''[paths]
train = "{train_path}"
dev = "{dev_path}"

[system]
gpu_allocator = null

[nlp]
lang = "{self.lang}"
pipeline = ["tok2vec", "ner"]
batch_size = 1000

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 2500, 2500, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${{components.tok2vec.model.encode.width}}

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 42
gpu_allocator = null
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
size = 1000

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
'''

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(config)

        return config


def validate_spans(data: List[Dict]) -> List[str]:
    """Validate all entity spans in the data.

    Args:
        data: List of annotation dicts.

    Returns:
        List of error messages (empty if all valid).
    """
    errors = []

    for i, item in enumerate(data):
        text = item.get("text", "")
        entities = item.get("entities", [])

        for j, ent in enumerate(entities):
            if len(ent) < 3:
                errors.append(f"Item {i}, entity {j}: Missing fields (need [start, end, label])")
                continue

            start, end, label = ent[0], ent[1], ent[2]

            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"Item {i}, entity {j}: start/end must be integers")
                continue

            if start < 0:
                errors.append(f"Item {i}, entity {j}: Negative start offset")

            if end > len(text):
                errors.append(f"Item {i}, entity {j}: End {end} exceeds text length {len(text)}")

            if start >= end:
                errors.append(f"Item {i}, entity {j}: start >= end ({start} >= {end})")

            if label not in {"PER", "LOC", "ORG", "MISC"}:
                errors.append(f"Item {i}, entity {j}: Unknown label '{label}'")

            # Check for overlaps within this item
            for k, other in enumerate(entities):
                if k <= j:
                    continue
                o_start, o_end = other[0], other[1]
                if start < o_end and end > o_start:
                    errors.append(
                        f"Item {i}: Overlapping entities [{start}:{end}] and [{o_start}:{o_end}]"
                    )

    return errors
