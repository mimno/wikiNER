"""Entity annotation by finding link positions in text."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..classifier.entity_classifier import EntityClassifier
from ..fetcher.api_client import WikiLink, WikiPage, WikipediaClient


@dataclass
class EntitySpan:
    """An entity span in text."""

    start: int  # Character offset start
    end: int  # Character offset end
    text: str  # The entity text
    label: str  # Entity type (PER, LOC, ORG, MISC)
    wiki_title: str  # Linked Wikipedia page title


class EntityAnnotator:
    """Annotate text with entity spans based on Wikipedia links."""

    def __init__(
        self,
        client: WikipediaClient,
        classifier: EntityClassifier,
        skip_misc: bool = False,
        fast_mode: bool = False,
    ):
        """Initialize the annotator.

        Args:
            client: Wikipedia client for fetching linked pages.
            classifier: Entity classifier instance.
            skip_misc: If True, don't include MISC entities in output.
            fast_mode: If True, skip fetching linked pages for classification.
        """
        self.client = client
        self.classifier = classifier
        self.skip_misc = skip_misc
        self.fast_mode = fast_mode

    def annotate_page(self, page: WikiPage) -> List[EntitySpan]:
        """Annotate a Wikipedia page with entity spans.

        Args:
            page: WikiPage object with text and links.

        Returns:
            List of EntitySpan objects sorted by start position.
        """
        if not page.exists or not page.text:
            return []

        entities = []
        seen_spans = set()  # Track (start, end) to avoid duplicates

        for link in page.links:
            # Find positions of this link's anchor text in the page text
            positions = self._find_positions(page.text, link.anchor)

            if not positions:
                continue

            # Classify the linked entity
            label = self._classify_link(link)

            if self.skip_misc and label == "MISC":
                continue

            # Create entity spans for each occurrence
            for start, end in positions:
                span_key = (start, end)
                if span_key in seen_spans:
                    continue

                # Check for overlapping entities - skip if overlap
                if self._has_overlap(span_key, seen_spans):
                    continue

                seen_spans.add(span_key)
                entities.append(
                    EntitySpan(
                        start=start,
                        end=end,
                        text=link.anchor,
                        label=label,
                        wiki_title=link.target,
                    )
                )

        # Sort by start position
        entities.sort(key=lambda e: e.start)
        return entities

    def _find_positions(
        self, text: str, anchor: str
    ) -> List[Tuple[int, int]]:
        """Find all positions of anchor text in the page.

        Uses word boundary matching to avoid partial matches.

        Args:
            text: Full page text.
            anchor: Anchor text to find.

        Returns:
            List of (start, end) tuples.
        """
        if not anchor or len(anchor) < 2:
            return []

        positions = []
        # Escape special regex characters in anchor
        escaped = re.escape(anchor)

        # Match with word boundaries for better precision
        # But also allow matches at start/end of text
        pattern = rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Only include if exact case match or anchor is a proper noun
            matched_text = match.group()
            if matched_text == anchor or anchor[0].isupper():
                positions.append((match.start(), match.end()))

        return positions

    def _classify_link(self, link: WikiLink) -> str:
        """Classify a linked entity.

        Args:
            link: WikiLink object.

        Returns:
            Entity type string.
        """
        # Fast mode: don't fetch linked pages, just use classifier directly
        if self.fast_mode:
            return self.classifier.classify(link.target)

        # Try to classify using the linked page
        try:
            page = self.client.get_page(link.target)
            if page.exists:
                return self.classifier.classify(
                    link.target,
                    categories=page.categories,
                    first_paragraph=None,  # Categories usually sufficient
                )
        except Exception:
            pass

        return "MISC"

    def _has_overlap(
        self, span: Tuple[int, int], existing: set
    ) -> bool:
        """Check if a span overlaps with existing spans.

        Args:
            span: (start, end) tuple to check.
            existing: Set of existing (start, end) tuples.

        Returns:
            True if there's an overlap.
        """
        start, end = span
        for ex_start, ex_end in existing:
            # Check for any overlap
            if start < ex_end and end > ex_start:
                return True
        return False

    def annotate_to_spacy_format(
        self, page: WikiPage
    ) -> Optional[Dict]:
        """Annotate a page and return in spaCy training format.

        Args:
            page: WikiPage object.

        Returns:
            Dict with "text" and "entities" keys, or None if no entities.
        """
        entities = self.annotate_page(page)

        if not entities:
            return None

        return {
            "text": page.text,
            "entities": [[e.start, e.end, e.label] for e in entities],
        }


def split_into_paragraphs(
    text: str, entities: List[EntitySpan]
) -> List[Dict]:
    """Split annotated text into paragraphs.

    This is useful because very long texts can be problematic for training.

    Args:
        text: Full page text.
        entities: List of entity spans.

    Returns:
        List of dicts, each with "text" and "entities" for one paragraph.
    """
    paragraphs = []
    current_pos = 0

    # Split on double newlines (paragraph breaks)
    for match in re.finditer(r"\n\n+", text):
        para_end = match.start()
        para_text = text[current_pos:para_end]

        if para_text.strip():
            # Find entities in this paragraph
            para_entities = []
            for e in entities:
                if current_pos <= e.start < para_end:
                    # Adjust offsets relative to paragraph start
                    para_entities.append(
                        [e.start - current_pos, e.end - current_pos, e.label]
                    )

            if para_entities:  # Only include paragraphs with entities
                paragraphs.append({"text": para_text, "entities": para_entities})

        current_pos = match.end()

    # Handle last paragraph
    if current_pos < len(text):
        para_text = text[current_pos:]
        if para_text.strip():
            para_entities = []
            for e in entities:
                if e.start >= current_pos:
                    para_entities.append(
                        [e.start - current_pos, e.end - current_pos, e.label]
                    )
            if para_entities:
                paragraphs.append({"text": para_text, "entities": para_entities})

    return paragraphs
