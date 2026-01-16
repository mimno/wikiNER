"""Command-line interface for WikiNER."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from .annotator.entity_annotator import EntityAnnotator, split_into_paragraphs
from .classifier.entity_classifier import EntityClassifier
from .exporter.json_exporter import JSONExporter
from .fetcher.api_client import WikipediaClient
from .fetcher.cache import WikiCache

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wiki_ner")


class ProcessingStats:
    """Track processing statistics."""

    def __init__(self):
        self.pages_processed = 0
        self.pages_skipped = 0
        self.pages_failed = 0
        self.total_entities = 0
        self.entities_by_type = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        self.end_time = datetime.now()

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def add_entities(self, entities):
        self.total_entities += len(entities)
        for e in entities:
            label = e.label.upper()
            if label in self.entities_by_type:
                self.entities_by_type[label] += 1

    def to_dict(self):
        return {
            "pages_processed": self.pages_processed,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "total_entities": self.total_entities,
            "entities_by_type": self.entities_by_type,
            "duration_seconds": self.duration,
        }

    def print_summary(self):
        click.echo("\nStatistics:")
        click.echo(f"  Pages processed: {self.pages_processed}")
        click.echo(f"  Pages skipped: {self.pages_skipped}")
        click.echo(f"  Pages failed: {self.pages_failed}")
        click.echo(f"  Total entities: {self.total_entities}")
        click.echo(
            f"  PER: {self.entities_by_type['PER']}, "
            f"LOC: {self.entities_by_type['LOC']}, "
            f"ORG: {self.entities_by_type['ORG']}, "
            f"MISC: {self.entities_by_type['MISC']}"
        )
        if self.duration > 0:
            click.echo(f"  Duration: {self.duration:.1f}s")


class ProgressState:
    """Save and restore processing progress for resuming."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.processed_urls = set()
        self.stats = ProcessingStats()

    def load(self) -> bool:
        """Load state from file. Returns True if state was loaded."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.processed_urls = set(data.get("processed_urls", []))
                return True
        return False

    def save(self):
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "processed_urls": list(self.processed_urls),
                    "stats": self.stats.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def mark_processed(self, url: str):
        """Mark a URL as processed."""
        self.processed_urls.add(url)

    def is_processed(self, url: str) -> bool:
        """Check if a URL has been processed."""
        return url in self.processed_urls

    def clear(self):
        """Clear saved state."""
        if self.state_file.exists():
            self.state_file.unlink()
        self.processed_urls.clear()


class _FastClassifier:
    """Dummy classifier that always returns MISC (for fast mode)."""

    def classify(self, title, categories=None, first_paragraph=None):
        return "MISC"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """WikiNER - Generate NER training data from Wikipedia."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output", default="output/train.json", help="Output JSON file path"
)
@click.option(
    "--cache-dir", default="./cache", help="Directory for caching API responses"
)
@click.option(
    "--skip-misc/--include-misc",
    default=False,
    help="Skip MISC entities in output",
)
@click.option(
    "--split-paragraphs/--no-split",
    default=True,
    help="Split pages into paragraphs",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Fast mode: skip entity classification (all marked as MISC)",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from previous progress",
)
@click.option(
    "--state-file",
    default=".wiki_ner_state.json",
    help="State file for resume capability",
)
@click.option(
    "--max-retries",
    default=3,
    help="Max retries per page on failure",
)
@click.option(
    "--stats-file",
    default=None,
    help="Export statistics to JSON file",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("-vv", "--debug", is_flag=True, help="Debug output (very verbose)")
def process(
    input_file: str,
    output: str,
    cache_dir: str,
    skip_misc: bool,
    split_paragraphs: bool,
    fast: bool,
    resume: bool,
    state_file: str,
    max_retries: int,
    stats_file: Optional[str],
    verbose: bool,
    debug: bool,
):
    """Process seed URLs and generate NER training data.

    INPUT_FILE: Text file with Wikipedia URLs (one per line).
    """
    # Set up logging level
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose:
        logger.setLevel(logging.INFO)

    # Read seed URLs
    urls = _read_seed_file(input_file)
    if not urls:
        click.echo("No valid URLs found in input file.", err=True)
        sys.exit(1)

    # Initialize progress state
    state = ProgressState(state_file)
    if resume and state.load():
        remaining = [u for u in urls if not state.is_processed(u)]
        click.echo(
            f"Resuming: {len(state.processed_urls)} already processed, "
            f"{len(remaining)} remaining"
        )
        urls = remaining
    else:
        state.clear()

    if not urls:
        click.echo("All URLs already processed.")
        return

    click.echo(f"Processing {len(urls)} Wikipedia pages...")
    if fast:
        click.echo("(Fast mode: entities will be marked as MISC)")

    # Initialize components
    cache = WikiCache(cache_dir)
    client = WikipediaClient(cache=cache)

    if fast:
        annotator = EntityAnnotator(
            client, _FastClassifier(), skip_misc=skip_misc, fast_mode=True
        )
    else:
        classifier = EntityClassifier(client=client, cache=cache)
        annotator = EntityAnnotator(client, classifier, skip_misc=skip_misc)

    exporter = JSONExporter()
    stats = state.stats
    stats.start()

    all_data = []

    # Load existing output if resuming
    output_path = Path(output)
    if resume and output_path.exists():
        try:
            all_data = JSONExporter.load(output)
            logger.info(f"Loaded {len(all_data)} existing examples from {output}")
        except Exception as e:
            logger.warning(f"Could not load existing output: {e}")

    # Process each URL with progress bar
    with tqdm(urls, desc="Processing pages", disable=debug) as pbar:
        for url in pbar:
            pbar.set_postfix_str(url.split("/")[-1][:20])

            success = _process_url_with_retry(
                url=url,
                client=client,
                annotator=annotator,
                stats=stats,
                all_data=all_data,
                split_paragraphs=split_paragraphs,
                max_retries=max_retries,
                verbose=verbose or debug,
            )

            # Mark as processed regardless of success (to avoid infinite retries)
            state.mark_processed(url)

            # Save progress periodically
            if stats.pages_processed % 10 == 0:
                state.save()
                # Also save partial output
                if all_data:
                    exporter.export(all_data, output)

    stats.stop()
    state.save()

    # Export final results
    if all_data:
        exporter.export(all_data, output)
        click.echo(f"\nWrote {len(all_data)} examples to {output}")
        stats.print_summary()

        # Export stats if requested
        if stats_file:
            with open(stats_file, "w") as f:
                json.dump(stats.to_dict(), f, indent=2)
            click.echo(f"Statistics saved to {stats_file}")
    else:
        click.echo("No training data generated.", err=True)
        sys.exit(1)

    # Clean up state file on successful completion
    if stats.pages_failed == 0:
        state.clear()
        click.echo("Processing complete. State file cleaned up.")


def _process_url_with_retry(
    url: str,
    client: WikipediaClient,
    annotator: EntityAnnotator,
    stats: ProcessingStats,
    all_data: list,
    split_paragraphs: bool,
    max_retries: int,
    verbose: bool,
) -> bool:
    """Process a single URL with retry logic.

    Returns True on success, False on failure.
    """
    for attempt in range(max_retries):
        try:
            page = client.get_page(url)
            if not page.exists:
                logger.info(f"Skipped (not found): {url}")
                stats.pages_skipped += 1
                return True  # Not a failure, just skip

            entities = annotator.annotate_page(page)
            if not entities:
                logger.info(f"Skipped (no entities): {url}")
                stats.pages_skipped += 1
                return True

            # Update stats
            stats.pages_processed += 1
            stats.add_entities(entities)

            # Convert to output format
            if split_paragraphs:
                paragraphs = split_into_paragraphs(page.text, entities)
                all_data.extend(paragraphs)
            else:
                all_data.append(
                    {
                        "text": page.text,
                        "entities": [[e.start, e.end, e.label] for e in entities],
                    }
                )

            logger.info(f"Processed: {page.title} ({len(entities)} entities)")
            return True

        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(
                f"Error processing {url} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

    logger.error(f"Failed after {max_retries} attempts: {url}")
    stats.pages_failed += 1
    return False


@cli.command()
@click.argument("url_or_title")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
@click.option("--explain", is_flag=True, help="Show classification explanation with signals")
def classify(url_or_title: str, verbose: bool, explain: bool):
    """Classify a single Wikipedia entity.

    URL_OR_TITLE: Wikipedia URL or page title.
    """
    cache = WikiCache("./cache")
    client = WikipediaClient(cache=cache)
    classifier = EntityClassifier(client=client, cache=cache)

    page = client.get_page(url_or_title)
    if not page.exists:
        click.echo(f"Page not found: {url_or_title}", err=True)
        sys.exit(1)

    # Get first paragraph for keyword classification
    first_para = client.get_first_paragraph(page.title)

    # Get detailed classification
    result = classifier.classify(
        page.title,
        categories=page.categories,
        first_paragraph=first_para,
        return_details=True,
    )

    click.echo(f"Title: {page.title}")
    click.echo(f"Classification: {result.label}")
    click.echo(f"Confidence: {result.confidence:.2f}")
    click.echo(f"Source: {result.source}")

    if result.is_disambiguation:
        click.echo("(Disambiguation page)")

    if verbose or explain:
        click.echo(f"\nCategories ({len(page.categories)}):")
        for cat in page.categories[:10]:
            click.echo(f"  - {cat}")
        if len(page.categories) > 10:
            click.echo(f"  ... and {len(page.categories) - 10} more")

    if explain:
        explanation = classifier.explain_classification(
            page.title, page.categories, first_para
        )

        click.echo(f"\nCategory signals:")
        for label in ["PER", "LOC", "ORG"]:
            signals = explanation.get("category_signals", {}).get(label, [])
            if signals:
                click.echo(f"  {label}: {len(signals)} matches")
                for sig in signals[:3]:
                    click.echo(f"    - {sig['category'][:50]}")

        click.echo(f"\nKeyword signals:")
        for label in ["PER", "LOC", "ORG"]:
            signals = explanation.get("keyword_signals", {}).get(label, [])
            if signals:
                click.echo(f"  {label}: {len(signals)} matches")
                for sig in signals[:3]:
                    click.echo(f"    - \"{sig['matched'][:40]}\"")


@cli.command()
@click.argument("url_or_title")
@click.option(
    "-o",
    "--output",
    default=None,
    help="Output JSON file (prints to stdout if not set)",
)
@click.option("--skip-misc/--include-misc", default=False, help="Skip MISC entities")
@click.option(
    "--max-entities", default=0, help="Limit number of entities (0=unlimited)"
)
@click.option(
    "--fast", is_flag=True, help="Fast mode: skip classification, mark all as MISC"
)
def annotate(
    url_or_title: str,
    output: Optional[str],
    skip_misc: bool,
    max_entities: int,
    fast: bool,
):
    """Annotate a single Wikipedia page.

    URL_OR_TITLE: Wikipedia URL or page title.
    """
    cache = WikiCache("./cache")
    client = WikipediaClient(cache=cache)

    if fast:
        # Fast mode: use a classifier that always returns MISC, skip fetching linked pages
        annotator = EntityAnnotator(
            client, _FastClassifier(), skip_misc=skip_misc, fast_mode=True
        )
    else:
        classifier = EntityClassifier(client=client, cache=cache)
        annotator = EntityAnnotator(client, classifier, skip_misc=skip_misc)

    page = client.get_page(url_or_title)
    if not page.exists:
        click.echo(f"Page not found: {url_or_title}", err=True)
        sys.exit(1)

    entities = annotator.annotate_page(page)
    if max_entities > 0:
        entities = entities[:max_entities]

    click.echo(f"Title: {page.title}")
    click.echo(f"Entities found: {len(entities)}\n")

    for e in entities[:20]:  # Show first 20
        click.echo(f'  [{e.start}:{e.end}] {e.label}: "{e.text}" -> {e.wiki_title}')

    if len(entities) > 20:
        click.echo(f"  ... and {len(entities) - 20} more")

    if output:
        exporter = JSONExporter()
        data = annotator.annotate_to_spacy_format(page)
        if data:
            exporter.export([data], output)
            click.echo(f"\nSaved to {output}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def validate(input_file: str):
    """Validate a JSON training data file.

    INPUT_FILE: JSON file to validate.
    """
    data = JSONExporter.load(input_file)
    errors = JSONExporter.validate(data)

    if errors:
        click.echo(f"Found {len(errors)} validation errors:", err=True)
        for error in errors[:10]:
            click.echo(f"  - {error}", err=True)
        if len(errors) > 10:
            click.echo(f"  ... and {len(errors) - 10} more", err=True)
        sys.exit(1)
    else:
        click.echo(f"Valid! {len(data)} examples, no errors found.")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir", default="output", help="Output directory for .spacy files"
)
@click.option(
    "--train-ratio", default=0.8, help="Proportion for training set (default 0.8)"
)
@click.option(
    "--dev-ratio", default=0.1, help="Proportion for dev set (default 0.1)"
)
@click.option(
    "--test-ratio", default=0.1, help="Proportion for test set (default 0.1)"
)
@click.option("--seed", default=42, help="Random seed for shuffling")
@click.option("--no-shuffle", is_flag=True, help="Don't shuffle before splitting")
@click.option("--config", is_flag=True, help="Generate spaCy training config file")
def convert(
    input_file: str,
    output_dir: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    no_shuffle: bool,
    config: bool,
):
    """Convert JSON annotations to spaCy training format.

    INPUT_FILE: JSON file with annotations.

    Creates train.spacy, dev.spacy, and test.spacy files in the output directory.
    """
    from .exporter.spacy_exporter import SpacyExporter

    exporter = SpacyExporter()

    click.echo(f"Converting {input_file} to spaCy format...")
    click.echo(f"Split: train={train_ratio}, dev={dev_ratio}, test={test_ratio}")

    try:
        results = exporter.convert_and_split(
            input_file,
            output_dir,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            shuffle=not no_shuffle,
            seed=seed,
        )

        click.echo(f"\nConversion complete:")
        for split_name, (success, skip) in results.items():
            output_path = Path(output_dir) / f"{split_name}.spacy"
            click.echo(f"  {split_name}: {success} docs ({skip} skipped) -> {output_path}")

        # Generate config if requested
        if config:
            config_path = Path(output_dir) / "config.cfg"
            train_path = Path(output_dir) / "train.spacy"
            dev_path = Path(output_dir) / "dev.spacy"

            exporter.generate_config(
                str(config_path),
                str(train_path),
                str(dev_path),
            )
            click.echo(f"\nGenerated training config: {config_path}")
            click.echo("\nTo train a model, run:")
            click.echo(f"  python -m spacy train {config_path} --output ./model")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--state-file",
    default=".wiki_ner_state.json",
    help="State file to show/clear",
)
@click.option("--clear", is_flag=True, help="Clear the saved state")
def status(state_file: str, clear: bool):
    """Show or clear processing status.

    Use this to check progress or clear state for a fresh start.
    """
    state = ProgressState(state_file)

    if clear:
        state.clear()
        click.echo("State cleared.")
        return

    if state.load():
        click.echo(f"State file: {state_file}")
        click.echo(f"URLs processed: {len(state.processed_urls)}")
        if state.processed_urls:
            click.echo("\nLast 5 processed:")
            for url in list(state.processed_urls)[-5:]:
                click.echo(f"  - {url}")
    else:
        click.echo("No saved state found.")


def _read_seed_file(path: str) -> list:
    """Read URLs from a seed file.

    Args:
        path: Path to seed file.

    Returns:
        List of URLs.
    """
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
