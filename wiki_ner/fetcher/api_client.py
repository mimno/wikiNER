"""Wikipedia API client for fetching page content, links, and categories."""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests
from ratelimit import limits, sleep_and_retry

from .cache import WikiCache


@dataclass
class WikiLink:
    """A Wikipedia internal link with anchor text and target."""

    target: str  # The linked page title
    anchor: str  # The display text in the article


@dataclass
class WikiPage:
    """Wikipedia page data."""

    title: str
    text: str  # Plain text content
    links: List[WikiLink]  # Internal links with anchors
    categories: List[str]  # Category names
    exists: bool


class WikipediaClient:
    """Client for fetching Wikipedia page data via the MediaWiki API."""

    API_URL = "https://en.wikipedia.org/w/api.php"
    USER_AGENT = "WikiNER/0.1 (NER data generation tool; contact@example.com)"

    def __init__(
        self,
        cache: Optional[WikiCache] = None,
        rate_limit: int = 1,
        language: str = "en",
    ):
        """Initialize the Wikipedia client.

        Args:
            cache: Optional cache instance for storing responses.
            rate_limit: Requests per second (default: 1).
            language: Wikipedia language code (default: "en").
        """
        self.cache = cache
        self.rate_limit = rate_limit
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    @sleep_and_retry
    @limits(calls=1, period=1)
    def _request(self, params: Dict) -> Dict:
        """Make a rate-limited API request.

        Args:
            params: API parameters.

        Returns:
            JSON response as dict.
        """
        params["format"] = "json"
        response = self.session.get(self.api_url, params=params)
        response.raise_for_status()
        return response.json()

    def _extract_title_from_url(self, url: str) -> str:
        """Extract page title from Wikipedia URL.

        Args:
            url: Full Wikipedia URL.

        Returns:
            Page title.
        """
        parsed = urlparse(url)
        path = parsed.path
        if "/wiki/" in path:
            title = path.split("/wiki/")[-1]
            return unquote(title).replace("_", " ")
        return url

    def get_page(self, url_or_title: str) -> WikiPage:
        """Fetch a Wikipedia page with text, links, and categories.

        Args:
            url_or_title: Wikipedia URL or page title.

        Returns:
            WikiPage with content, links, and categories.
        """
        # Extract title from URL if needed
        if url_or_title.startswith("http"):
            title = self._extract_title_from_url(url_or_title)
        else:
            title = url_or_title

        # Check cache first
        cache_key = f"page:{title}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # Fetch page content
        text = self._get_page_text(title)
        if text is None:
            return WikiPage(
                title=title, text="", links=[], categories=[], exists=False
            )

        # Fetch links with anchors
        links = self._get_page_links_with_anchors(title)

        # Fetch categories
        categories = self._get_page_categories(title)

        page = WikiPage(
            title=title, text=text, links=links, categories=categories, exists=True
        )

        # Cache the result
        if self.cache:
            self.cache.set(cache_key, page)

        return page

    def _get_page_text(self, title: str) -> Optional[str]:
        """Get plain text content of a page.

        Args:
            title: Page title.

        Returns:
            Plain text or None if page doesn't exist.
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": "1",  # Plain text, no HTML
            "exsectionformat": "plain",
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})

        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")

        return None

    def _get_page_links_with_anchors(self, title: str) -> List[WikiLink]:
        """Get internal links with their anchor text.

        Uses the parse API to get wikitext and extract link patterns.

        Args:
            title: Page title.

        Returns:
            List of WikiLink objects.
        """
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
        }
        try:
            data = self._request(params)
        except requests.HTTPError:
            return []

        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")

        # Extract [[target|anchor]] or [[target]] patterns
        links = []
        # Match [[...]] but not [[File:...]], [[Image:...]], [[Category:...]]
        pattern = r"\[\[(?!File:|Image:|Category:|Wikipedia:)([^\]|]+)(?:\|([^\]]+))?\]\]"

        for match in re.finditer(pattern, wikitext):
            target = match.group(1).strip()
            anchor = match.group(2).strip() if match.group(2) else target
            # Skip section links
            if "#" in target:
                target = target.split("#")[0]
            if target:
                links.append(WikiLink(target=target, anchor=anchor))

        return links

    def _get_page_categories(self, title: str) -> List[str]:
        """Get categories for a page.

        Args:
            title: Page title.

        Returns:
            List of category names (without "Category:" prefix).
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "categories",
            "cllimit": "500",
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})

        categories = []
        for page_data in pages.values():
            for cat in page_data.get("categories", []):
                cat_title = cat.get("title", "")
                # Remove "Category:" prefix
                if cat_title.startswith("Category:"):
                    cat_title = cat_title[9:]
                categories.append(cat_title)

        return categories

    def get_first_paragraph(self, title: str) -> str:
        """Get just the first paragraph of a page.

        Useful for keyword-based classification.

        Args:
            title: Page title.

        Returns:
            First paragraph text.
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": "1",
            "exintro": "1",  # Only the intro section
        }
        data = self._request(params)
        pages = data.get("query", {}).get("pages", {})

        for page_data in pages.values():
            return page_data.get("extract", "")

        return ""
