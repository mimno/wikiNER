"""Combined entity classifier using categories and keywords."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..fetcher.api_client import WikipediaClient
from ..fetcher.cache import WikiCache
from .category_rules import classify_by_category, ClassificationResult, get_category_signals
from .keyword_rules import classify_by_keywords, KeywordClassificationResult, get_keyword_signals


@dataclass
class EntityClassification:
    """Complete classification result with confidence and reasoning."""

    label: str
    confidence: float
    source: str  # "category", "keyword", "combined", or "default"
    category_matches: int
    keyword_matches: int
    is_disambiguation: bool


# Patterns that indicate a disambiguation page
DISAMBIGUATION_PATTERNS = [
    r"disambiguation",
    r"may refer to",
    r"can refer to",
    r"commonly refers to",
    r"most commonly refers to",
    r"list of people",
    r"list of places",
    r"index of articles",
]


class EntityClassifier:
    """Classify Wikipedia entities into PER, LOC, ORG, or MISC.

    Uses a two-stage approach:
    1. Primary: Category-based classification (most reliable)
    2. Fallback: Keyword-based classification from first paragraph

    Also handles disambiguation pages specially.
    """

    def __init__(
        self,
        client: Optional[WikipediaClient] = None,
        cache: Optional[WikiCache] = None,
        min_confidence: float = 0.0,
    ):
        """Initialize the classifier.

        Args:
            client: Wikipedia client for fetching page data.
            cache: Optional cache for classification results.
            min_confidence: Minimum confidence threshold (below returns MISC).
        """
        self.client = client
        self.cache = cache
        self.min_confidence = min_confidence
        self._classification_cache: Dict[str, str] = {}

    def classify(
        self,
        title: str,
        categories: Optional[List[str]] = None,
        first_paragraph: Optional[str] = None,
        return_details: bool = False,
    ) -> str | EntityClassification:
        """Classify an entity by its Wikipedia page.

        Args:
            title: Wikipedia page title.
            categories: Pre-fetched categories (optional).
            first_paragraph: Pre-fetched first paragraph (optional).
            return_details: If True, return EntityClassification with full details.

        Returns:
            Entity type: "PER", "LOC", "ORG", or "MISC".
            If return_details is True, returns EntityClassification object.
        """
        # Check in-memory cache (simple label only)
        if not return_details and title in self._classification_cache:
            return self._classification_cache[title]

        # Check disk cache
        cache_key = f"entity_type:{title}"
        if not return_details and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                self._classification_cache[title] = cached
                return cached

        # Check for disambiguation page
        is_disambiguation = self._is_disambiguation(title, categories, first_paragraph)
        if is_disambiguation:
            result = EntityClassification(
                label="MISC",
                confidence=1.0,
                source="disambiguation",
                category_matches=0,
                keyword_matches=0,
                is_disambiguation=True,
            )
            if return_details:
                return result
            self._cache_result(title, cache_key, "MISC")
            return "MISC"

        # Stage 1: Category-based classification
        category_result = None
        category_matches = 0
        if categories is not None:
            category_result = classify_by_category(categories, return_confidence=True)
            if category_result and category_result.label != "MISC":
                category_matches = len(category_result.matched_patterns)

        # Stage 2: Keyword-based classification
        keyword_result = None
        keyword_matches = 0
        if first_paragraph is not None:
            keyword_result = classify_by_keywords(first_paragraph, return_confidence=True)
            if keyword_result and keyword_result.label != "MISC":
                keyword_matches = len(keyword_result.matches)

        # Combine results
        final_label, final_confidence, source = self._combine_results(
            category_result, keyword_result
        )

        # Apply minimum confidence threshold
        if final_confidence < self.min_confidence:
            final_label = "MISC"
            source = "default"

        # If we have a client but no pre-fetched data, try fetching
        if (
            self.client
            and final_label == "MISC"
            and (categories is None or first_paragraph is None)
        ):
            fetched_result = self._classify_by_fetching(title)
            if fetched_result and fetched_result.label != "MISC":
                final_label = fetched_result.label
                final_confidence = fetched_result.confidence
                source = fetched_result.source
                category_matches = fetched_result.category_matches
                keyword_matches = fetched_result.keyword_matches

        # Cache the result
        self._cache_result(title, cache_key, final_label)

        if return_details:
            return EntityClassification(
                label=final_label,
                confidence=final_confidence,
                source=source,
                category_matches=category_matches,
                keyword_matches=keyword_matches,
                is_disambiguation=False,
            )

        return final_label

    def _is_disambiguation(
        self,
        title: str,
        categories: Optional[List[str]],
        first_paragraph: Optional[str],
    ) -> bool:
        """Check if a page is a disambiguation page.

        Args:
            title: Page title.
            categories: Page categories.
            first_paragraph: First paragraph text.

        Returns:
            True if this appears to be a disambiguation page.
        """
        # Check title
        if "disambiguation" in title.lower():
            return True

        # Check categories
        if categories:
            for cat in categories:
                if "disambiguation" in cat.lower():
                    return True

        # Check first paragraph
        if first_paragraph:
            text_lower = first_paragraph.lower()
            for pattern in DISAMBIGUATION_PATTERNS:
                if re.search(pattern, text_lower):
                    return True

        return False

    def _combine_results(
        self,
        category_result: Optional[ClassificationResult],
        keyword_result: Optional[KeywordClassificationResult],
    ) -> Tuple[str, float, str]:
        """Combine category and keyword results.

        Args:
            category_result: Result from category classification.
            keyword_result: Result from keyword classification.

        Returns:
            Tuple of (label, confidence, source).
        """
        cat_label = category_result.label if category_result else "MISC"
        cat_conf = category_result.confidence if category_result else 0.0

        kw_label = keyword_result.label if keyword_result else "MISC"
        kw_conf = keyword_result.confidence if keyword_result else 0.0

        # If both agree, boost confidence
        if cat_label == kw_label and cat_label != "MISC":
            combined_conf = min(cat_conf + kw_conf * 0.3, 1.0)
            return cat_label, combined_conf, "combined"

        # Category has priority (more reliable)
        if cat_label != "MISC" and cat_conf >= 0.5:
            return cat_label, cat_conf, "category"

        # Fall back to keywords if confident
        if kw_label != "MISC" and kw_conf >= 0.5:
            return kw_label, kw_conf, "keyword"

        # Use whichever has higher confidence
        if cat_conf > kw_conf and cat_label != "MISC":
            return cat_label, cat_conf, "category"
        if kw_conf > cat_conf and kw_label != "MISC":
            return kw_label, kw_conf, "keyword"

        return "MISC", 0.0, "default"

    def _classify_by_fetching(self, title: str) -> Optional[EntityClassification]:
        """Classify by fetching page data from Wikipedia.

        Args:
            title: Wikipedia page title.

        Returns:
            EntityClassification or None on failure.
        """
        if not self.client:
            return None

        try:
            page = self.client.get_page(title)
            if not page.exists:
                return None

            # Check disambiguation
            if self._is_disambiguation(title, page.categories, None):
                return EntityClassification(
                    label="MISC",
                    confidence=1.0,
                    source="disambiguation",
                    category_matches=0,
                    keyword_matches=0,
                    is_disambiguation=True,
                )

            # Try category-based
            category_result = classify_by_category(page.categories, return_confidence=True)

            # Try keyword-based with first paragraph
            first_para = self.client.get_first_paragraph(title)
            keyword_result = classify_by_keywords(first_para, return_confidence=True)

            label, confidence, source = self._combine_results(category_result, keyword_result)

            return EntityClassification(
                label=label,
                confidence=confidence,
                source=source,
                category_matches=len(category_result.matched_patterns) if category_result else 0,
                keyword_matches=len(keyword_result.matches) if keyword_result else 0,
                is_disambiguation=False,
            )

        except Exception:
            return None

    def _cache_result(self, title: str, cache_key: str, result: str) -> None:
        """Cache a classification result.

        Args:
            title: Wikipedia page title.
            cache_key: Disk cache key.
            result: Classification result.
        """
        self._classification_cache[title] = result
        if self.cache:
            self.cache.set(cache_key, result)

    def classify_batch(
        self,
        titles: List[str],
        return_details: bool = False,
    ) -> Dict[str, str] | Dict[str, EntityClassification]:
        """Classify multiple entities.

        Args:
            titles: List of Wikipedia page titles.
            return_details: If True, return EntityClassification objects.

        Returns:
            Dict mapping titles to entity types or classifications.
        """
        results = {}
        for title in titles:
            results[title] = self.classify(title, return_details=return_details)
        return results

    def explain_classification(
        self,
        title: str,
        categories: Optional[List[str]] = None,
        first_paragraph: Optional[str] = None,
    ) -> dict:
        """Get detailed explanation of classification decision.

        Useful for debugging and understanding why a classification was made.

        Args:
            title: Wikipedia page title.
            categories: Page categories.
            first_paragraph: First paragraph text.

        Returns:
            Dict with detailed classification signals.
        """
        result = self.classify(
            title, categories, first_paragraph, return_details=True
        )

        explanation = {
            "title": title,
            "classification": result.label,
            "confidence": result.confidence,
            "source": result.source,
            "is_disambiguation": result.is_disambiguation,
        }

        if categories:
            explanation["category_signals"] = get_category_signals(categories)

        if first_paragraph:
            explanation["keyword_signals"] = get_keyword_signals(first_paragraph)

        return explanation
