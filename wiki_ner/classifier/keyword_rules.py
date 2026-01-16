"""Keyword-based entity classification from first paragraph.

This module provides fallback classification when category-based
classification is insufficient. It analyzes the first paragraph
of Wikipedia articles for characteristic patterns.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

# =============================================================================
# PERSON keywords - Patterns indicating a person/individual
# =============================================================================

PERSON_KEYWORDS = [
    # Biographical patterns (strongest signals)
    r"\bwas born\b",
    r"\bborn .{0,30}\d{4}\b",
    r"\bdied .{0,30}\d{4}\b",
    r"\b\(\d{4}\s*[-–]\s*\d{4}\)",  # (1879-1955) lifespan
    r"\b\(\d{4}\s*[-–]\s*(present)?\)",  # (1950-) or (1950-present)
    r"\bborn in\b",
    r"\braised in\b",
    r"\bgrew up in\b",

    # Personal pronouns (very strong signal for people)
    r"\b(he|she) (is|was|has|had|became|worked|served|played|won|received)\b",
    r"\b(his|her) (life|career|work|family|father|mother|parents|childhood|education)\b",
    r"\b(himself|herself)\b",

    # Occupational patterns
    r"\b(is|was) an? .{0,40}\b(person|man|woman|human being)\b",
    r"\b(is|was) an? .{0,40}\b(actor|actress|singer|musician|artist|writer|author)\b",
    r"\b(is|was) an? .{0,40}\b(politician|president|prime minister|senator|governor)\b",
    r"\b(is|was) an? .{0,40}\b(scientist|physicist|chemist|biologist|mathematician)\b",
    r"\b(is|was) an? .{0,40}\b(athlete|player|footballer|basketball player)\b",
    r"\b(is|was) an? .{0,40}\b(director|producer|filmmaker|screenwriter)\b",
    r"\b(is|was) an? .{0,40}\b(journalist|reporter|broadcaster|anchor)\b",
    r"\b(is|was) an? .{0,40}\b(businessman|businesswoman|entrepreneur|executive|CEO)\b",
    r"\b(is|was) an? .{0,40}\b(lawyer|attorney|judge|justice)\b",
    r"\b(is|was) an? .{0,40}\b(doctor|physician|surgeon|nurse)\b",
    r"\b(is|was) an? .{0,40}\b(professor|teacher|academic|scholar)\b",
    r"\b(is|was) an? .{0,40}\b(admiral|commander|soldier|military officer|military leader)\b",
    r"\bgeneral\b.{0,20}\b(of the|in the)\b",  # "General of the Army"
    r"\b(is|was) an? .{0,40}\b(philosopher|historian|economist|sociologist)\b",
    r"\b(is|was) an? .{0,40}\b(chef|cook|restaurateur)\b",
    r"\b(is|was) an? .{0,40}\b(comedian|humorist|satirist)\b",
    r"\b(is|was) an? .{0,40}\b(model|fashion model|supermodel)\b",
    r"\b(is|was) an? .{0,40}\b(architect|designer|engineer|inventor)\b",

    # Career patterns
    r"\bcareer\b.{0,30}\b(began|started|spanning|lasted)\b",
    r"\bknown for (his|her)\b",
    r"\bbest known for\b",
    r"\bfamous for\b",
    r"\brose to (fame|prominence)\b",
    r"\bwon .{0,20}(award|prize|medal|championship)\b",
    r"\breceived .{0,20}(award|prize|honor|recognition)\b",
    r"\bnobel (prize|laureate)\b",

    # Educational/training patterns
    r"\bstudied at\b",
    r"\bgraduated from\b",
    r"\battended .{0,30}(university|college|school)\b",
    r"\beducated at\b",

    # Relationship patterns
    r"\bmarried to\b",
    r"\bmarried .{0,30}in \d{4}\b",
    r"\b(his|her) (wife|husband|spouse|partner|children|son|daughter)\b",
    r"\bfather of\b",
    r"\bmother of\b",

    # Title patterns
    r"\b(sir|dame|lord|lady|dr\.|prof\.|rev\.)\s+[A-Z]",
    r"\bking\s+[A-Z]",
    r"\bqueen\s+[A-Z]",
    r"\bprince\s+[A-Z]",
    r"\bprincess\s+[A-Z]",
    r"\bpope\s+[A-Z]",
    r"\bsaint\s+[A-Z]",
]

# =============================================================================
# LOCATION keywords - Patterns indicating a place/location
# =============================================================================

LOCATION_KEYWORDS = [
    # Place type patterns (strongest signals)
    r"\b(is|was) an? .{0,30}\b(city|town|village|hamlet)\b",
    r"\b(is|was) an? .{0,30}\b(country|nation|state|republic|kingdom|empire)\b",
    r"\b(is|was) an? .{0,30}\b(region|territory|province|district|county)\b",
    r"\b(is|was) an? .{0,30}\b(island|peninsula|archipelago|atoll)\b",
    r"\b(is|was) an? .{0,30}\b(mountain|hill|peak|volcano|ridge)\b",
    r"\b(is|was) an? .{0,30}\b(river|lake|sea|ocean|bay|gulf|strait)\b",
    r"\b(is|was) an? .{0,30}\b(valley|plain|plateau|desert|forest)\b",
    r"\b(is|was) an? .{0,30}\b(municipality|commune|borough|township)\b",
    r"\b(is|was) the capital\b",
    r"\b(is|was) the largest city\b",

    # Geographic location patterns
    r"\bis located in\b",
    r"\bis situated in\b",
    r"\blies (in|on|along|near)\b",
    r"\bfound in .{0,30}(region|area|part)\b",
    r"\bin the .{0,30}(region|area|part) of\b",
    r"\bborders\b.{0,30}\b(country|state|region|river|sea)\b",
    r"\bsurrounded by\b",
    r"\bon the (coast|shore|bank|border) of\b",
    r"\bat the (mouth|source|confluence) of\b",

    # Population patterns
    r"\bpopulation of\b",
    r"\bpopulation .{0,20}\b\d{1,3}(,\d{3})*\b",
    r"\b\d{1,3}(,\d{3})* (people|inhabitants|residents)\b",
    r"\bpopulated place\b",
    r"\bmost populous\b",

    # Geographic features
    r"\bcoordinates\b",
    r"\blatitude\b",
    r"\blongitude\b",
    r"\belevation\b",
    r"\barea of .{0,20}(km|square|sq)\b",
    r"\bsquare (kilometers|kilometres|miles)\b",
    r"\bkm²\b",
    r"\bmi²\b",

    # Administrative patterns
    r"\bcapital (of|city)\b",
    r"\bseat of\b",
    r"\badministrative (center|centre|capital)\b",
    r"\bpart of the .{0,30}(region|province|state|country)\b",
    r"\bincorporated (in|as)\b",
    r"\bfounded in \d{4}\b.{0,30}(city|town|settlement)",

    # Geographic descriptors
    r"\bnorthern\b.{0,30}\b(region|part|area)\b",
    r"\bsouthern\b.{0,30}\b(region|part|area)\b",
    r"\beastern\b.{0,30}\b(region|part|area)\b",
    r"\bwestern\b.{0,30}\b(region|part|area)\b",
    r"\bcentral\b.{0,30}\b(region|part|area)\b",

    # Climate/environment
    r"\bclimate\b.{0,30}\b(tropical|temperate|arid|continental)\b",
    r"\btime zone\b",
]

# =============================================================================
# ORGANIZATION keywords - Patterns indicating an organization/company
# =============================================================================

ORGANIZATION_KEYWORDS = [
    # Organization type patterns (strongest signals)
    r"\b(is|was) an? .{0,40}\b(company|corporation|firm|enterprise)\b",
    r"\b(is|was) an? .{0,40}\b(organization|organisation|institution|agency)\b",
    r"\b(is|was) an? .{0,40}\b(university|college|school|academy|institute)\b",
    r"\b(is|was) an? .{0,40}\b(bank|financial institution|insurance company)\b",
    r"\b(is|was) an? .{0,40}\b(newspaper|magazine|publisher|media company)\b",
    r"\b(is|was) an? .{0,40}\b(team|club|squad|franchise)\b",
    r"\b(is|was) an? .{0,40}\b(airline|carrier|railway|railroad)\b",
    r"\b(is|was) an? .{0,40}\b(church|temple|mosque|religious organization)\b",
    r"\b(is|was) an? .{0,40}\b(hospital|clinic|medical center|healthcare)\b",
    r"\b(is|was) an? .{0,40}\b(museum|gallery|library|archive)\b",
    r"\b(is|was) an? .{0,40}\b(political party|movement|coalition)\b",
    r"\b(is|was) an? .{0,40}\b(band|group|ensemble|orchestra|choir)\b",
    r"\b(is|was) an? .{0,40}\b(charity|foundation|nonprofit|non-profit|NGO)\b",
    r"\b(is|was) an? .{0,40}\b(union|association|federation|league)\b",
    r"\b(is|was) an? .{0,40}\b(government agency|ministry|department)\b",

    # Business patterns
    r"\bfounded (in|by)\b",
    r"\bestablished (in|by)\b",
    r"\bincorporated (in|as)\b",
    r"\bheadquartered (in|at)\b",
    r"\bbased in\b.{0,30}\b(city|country|state)\b",
    r"\boperates\b",
    r"\bprovides\b.{0,30}\b(services|products)\b",
    r"\bmanufactures\b",
    r"\bproduces\b",
    r"\bsells\b",
    r"\bemploys\b.{0,20}\b\d",
    r"\bemployees\b",
    r"\bworkforce\b",
    r"\bstaff of\b",

    # Corporate structure
    r"\bsubsidiary of\b",
    r"\bparent company\b",
    r"\bholding company\b",
    r"\baffiliate of\b",
    r"\bmerged with\b",
    r"\bacquired by\b",
    r"\bacquired .{0,30}in \d{4}\b",
    r"\bpublicly traded\b",
    r"\bprivately held\b",
    r"\bstock exchange\b",
    r"\bticker symbol\b",
    r"\bNYSE\b",
    r"\bNASDAQ\b",

    # Financial metrics
    r"\brevenue of\b",
    r"\bmarket (cap|capitalization)\b",
    r"\bassets of\b",
    r"\bnet (income|worth)\b",
    r"\bprofit\b",
    r"\b(billion|million) (dollars|USD|EUR|GBP)\b",

    # Industry sectors
    r"\b(technology|tech) company\b",
    r"\bsoftware company\b",
    r"\bretail(er)?\b",
    r"\bmanufacturer\b",
    r"\bconglomerate\b",
    r"\bmultinational\b",

    # Sports organizations
    r"\bprofessional .{0,20}(team|club|franchise)\b",
    r"\bplays in\b.{0,30}\b(league|division|conference)\b",
    r"\bcompetes in\b",
    r"\bhome (stadium|arena|ground|field)\b",
    r"\bfounded .{0,20}(club|team)\b",

    # Educational organizations
    r"\bpublic (university|college|school)\b",
    r"\bprivate (university|college|school)\b",
    r"\benrolled students\b",
    r"\benrollment of\b",
    r"\bcampus\b",
    r"\baccredited\b",

    # Government organizations
    r"\bgovernment\b.{0,30}\b(agency|department|ministry)\b",
    r"\bfederal\b.{0,20}\b(agency|department)\b",
    r"\bstate\b.{0,20}\b(agency|department)\b",
    r"\bregulatory\b",
]


@dataclass
class KeywordMatch:
    """A keyword match with its context."""

    pattern: str
    matched_text: str
    position: int


@dataclass
class KeywordClassificationResult:
    """Result of keyword-based classification."""

    label: str
    confidence: float
    matches: List[KeywordMatch]


def classify_by_keywords(
    first_paragraph: str,
    return_confidence: bool = False
) -> Optional[str] | KeywordClassificationResult:
    """Classify entity type based on first paragraph keywords.

    Args:
        first_paragraph: First paragraph or intro of the Wikipedia page.
        return_confidence: If True, return KeywordClassificationResult.

    Returns:
        If return_confidence is False: "PER", "LOC", "ORG", or None if no match.
        If return_confidence is True: KeywordClassificationResult object.
    """
    if not first_paragraph:
        if return_confidence:
            return KeywordClassificationResult("MISC", 0.0, [])
        return None

    text = first_paragraph.lower()

    # Focus on first two sentences for better signal
    sentences = text.split(".")
    focus_text = ". ".join(sentences[:2]) if len(sentences) > 1 else text

    # Collect matches for each type
    matches = {
        "PER": [],
        "LOC": [],
        "ORG": [],
    }

    for pattern in PERSON_KEYWORDS:
        match = re.search(pattern, focus_text, re.IGNORECASE)
        if match:
            matches["PER"].append(KeywordMatch(
                pattern=pattern,
                matched_text=match.group(),
                position=match.start()
            ))

    for pattern in LOCATION_KEYWORDS:
        match = re.search(pattern, focus_text, re.IGNORECASE)
        if match:
            matches["LOC"].append(KeywordMatch(
                pattern=pattern,
                matched_text=match.group(),
                position=match.start()
            ))

    for pattern in ORGANIZATION_KEYWORDS:
        match = re.search(pattern, focus_text, re.IGNORECASE)
        if match:
            matches["ORG"].append(KeywordMatch(
                pattern=pattern,
                matched_text=match.group(),
                position=match.start()
            ))

    # Determine winner
    match_counts = {k: len(v) for k, v in matches.items()}
    max_matches = max(match_counts.values())

    if max_matches == 0:
        if return_confidence:
            return KeywordClassificationResult("MISC", 0.0, [])
        return None

    # Find the type with most matches
    winner = max(match_counts, key=match_counts.get)

    # Calculate confidence
    # Keywords in first sentence are stronger signals
    first_sentence = sentences[0] if sentences else text
    early_matches = sum(1 for m in matches[winner] if m.position < len(first_sentence))

    base_confidence = min(0.4 + (max_matches * 0.1), 0.8)
    early_bonus = min(early_matches * 0.05, 0.15)
    confidence = min(base_confidence + early_bonus, 0.95)

    if return_confidence:
        return KeywordClassificationResult(winner, confidence, matches[winner])

    return winner


def get_keyword_signals(text: str) -> dict:
    """Analyze text and return keyword signals for debugging.

    Args:
        text: Text to analyze (typically first paragraph).

    Returns:
        Dict with matches for each entity type.
    """
    if not text:
        return {"PER": [], "LOC": [], "ORG": []}

    text_lower = text.lower()
    signals = {"PER": [], "LOC": [], "ORG": []}

    for pattern in PERSON_KEYWORDS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            signals["PER"].append({
                "pattern": pattern,
                "matched": match.group(),
                "position": match.start()
            })

    for pattern in LOCATION_KEYWORDS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            signals["LOC"].append({
                "pattern": pattern,
                "matched": match.group(),
                "position": match.start()
            })

    for pattern in ORGANIZATION_KEYWORDS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            signals["ORG"].append({
                "pattern": pattern,
                "matched": match.group(),
                "position": match.start()
            })

    return signals
