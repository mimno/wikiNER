"""Category-based entity classification rules.

This module contains regex patterns for classifying Wikipedia entities
based on their category memberships. Patterns are ordered by specificity
and reliability.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# =============================================================================
# PERSON patterns - People, individuals, biographical subjects
# =============================================================================

PERSON_PATTERNS = [
    # Biographical markers (strongest signals)
    r"living people",
    r"\d{4} births",
    r"\d{4} deaths",
    r"\d{1,2}th-century .* (men|women|people)",
    r"\d{1,2}st-century .* (men|women|people)",
    r"people from ",
    r"people of .* descent",
    r"alumni of",
    r"(male|female) ",

    # Professions - Entertainment
    r" (actors|actresses)$",
    r" (film|television|stage|voice) actors",
    r" (singers|vocalists|rappers)$",
    r" (musicians|guitarists|pianists|drummers|bassists|violinists)$",
    r" (composers|songwriters|lyricists)$",
    r" (directors|filmmakers|screenwriters)$",
    r" (producers|film producers|record producers)$",
    r" (comedians|stand-up comedians)$",
    r" (dancers|choreographers|ballerinas)$",
    r" models$",
    r" (disc jockeys|DJs)$",

    # Professions - Writers & Arts
    r" (writers|authors|novelists|poets)$",
    r" (journalists|reporters|columnists|broadcasters)$",
    r" (editors|publishers)$",
    r" (artists|painters|sculptors|illustrators)$",
    r" (photographers|cinematographers)$",
    r" (architects|designers)$",
    r" (playwrights|dramatists)$",

    # Professions - Science & Academia
    r" (scientists|researchers)$",
    r" (physicists|chemists|biologists|mathematicians)$",
    r" (astronomers|astrophysicists|cosmologists)$",
    r" (geologists|meteorologists|oceanographers)$",
    r" (psychologists|sociologists|anthropologists)$",
    r" (economists|political scientists)$",
    r" (historians|archaeologists)$",
    r" (philosophers|theologians|ethicists)$",
    r" (engineers|inventors)$",
    r" (computer scientists|programmers)$",
    r" (linguists|philologists)$",
    r" (professors|academics|scholars)$",
    r" (physicians|doctors|surgeons|medical doctors)$",
    r" (nurses|pharmacists|dentists)$",

    # Professions - Politics & Government
    r" (politicians|statesmen|stateswomen)$",
    r" (presidents|vice presidents)$",
    r" (prime ministers|premiers|chancellors)$",
    r" (governors|lieutenant governors)$",
    r" (senators|congressmen|congresswomen|representatives)$",
    r" (mayors|city councillors)$",
    r" (ambassadors|diplomats|consuls)$",
    r" (monarchs|kings|queens|emperors|empresses)$",
    r" (princes|princesses|dukes|duchesses)$",
    r" (dictators|autocrats)$",
    r" (revolutionaries|rebels|activists)$",
    r" (judges|justices|magistrates)$",
    r" (lawyers|attorneys|barristers|solicitors)$",

    # Professions - Military
    r" (generals|admirals|commanders)$",
    r" (soldiers|warriors|veterans)$",
    r" (officers|military personnel)$",
    r" (pilots|aviators|astronauts|cosmonauts)$",
    r" (spies|intelligence officers)$",

    # Professions - Sports
    r" (athletes|sportspeople|sportsmen|sportswomen)$",
    r" (footballers|soccer players)$",
    r" (basketball players|NBA players)$",
    r" (baseball players|MLB players)$",
    r" (hockey players|NHL players)$",
    r" (tennis players|golfers|boxers|wrestlers)$",
    r" (swimmers|divers|water polo players)$",
    r" (runners|sprinters|marathon runners)$",
    r" (cyclists|racing drivers|motorsport)$",
    r" (skiers|snowboarders|figure skaters)$",
    r" (gymnasts|weightlifters)$",
    r" (cricket players|cricketers|rugby players)$",
    r" (coaches|managers|trainers)$",

    # Professions - Business
    r" (businesspeople|entrepreneurs|executives)$",
    r" (CEOs|chief executive officers)$",
    r" (philanthropists|investors)$",
    r" (bankers|financiers)$",

    # Professions - Religion
    r" (clergy|priests|ministers|pastors)$",
    r" (bishops|archbishops|cardinals|popes)$",
    r" (rabbis|imams|monks|nuns)$",
    r" (saints|beatified people)$",
    r" (missionaries|evangelists)$",

    # Professions - Other
    r" (chefs|cooks|restaurateurs)$",
    r" (explorers|adventurers|mountaineers)$",
    r" (criminals|outlaws|gangsters|serial killers)$",
    r" (magicians|illusionists)$",

    # Awards and recognition
    r"nobel laureates",
    r"recipients of ",
    r"members of the ",
    r"fellows of ",
    r"(grammy|emmy|oscar|tony|bafta) (award )?(winners|nominees)",
    r"pulitzer prize (winners|recipients)",
    r"order of ",
    r"knight",
    r"dame",

    # Nationalities with occupations
    r"(american|british|french|german|italian|spanish|russian|chinese|japanese|indian|australian|canadian) (people|men|women)",
]

# =============================================================================
# LOCATION patterns - Places, geographic features, administrative regions
# =============================================================================

LOCATION_PATTERNS = [
    # Administrative divisions
    r"^cities in ",
    r"^cities and towns in ",
    r"^towns in ",
    r"^villages in ",
    r"^municipalities in ",
    r"^municipalities of ",
    r"^communes in ",
    r"^communes of ",
    r"^counties in ",
    r"^counties of ",
    r"^districts in ",
    r"^districts of ",
    r"^provinces of ",
    r"^provinces in ",
    r"^states of ",
    r"^states and territories of ",
    r"^regions of ",
    r"^regions in ",
    r"^departments of ",
    r"^prefectures in ",
    r"^oblasts of ",
    r"^cantons of ",
    r"^parishes in ",
    r"^boroughs ",
    r"^neighborhoods in ",
    r"^suburbs of ",

    # Countries and capitals
    r"^countries in ",
    r"^countries of ",
    r"^capitals in ",
    r"^capitals of ",
    r"^capital cities",
    r"^sovereign states",
    r"^member states of ",
    r"^former countries",
    r"^landlocked countries",
    r"^island countries",

    # Geographic features - Water
    r"^rivers of ",
    r"^rivers in ",
    r"^lakes of ",
    r"^lakes in ",
    r"^seas of ",
    r"^bays of ",
    r"^gulfs of ",
    r"^straits of ",
    r"^channels of ",
    r"^fjords of ",
    r"^waterfalls ",
    r"^springs of ",
    r"^reservoirs in ",
    r"^bodies of water",
    r"^tributaries of ",

    # Geographic features - Land
    r"^mountains of ",
    r"^mountains in ",
    r"^mountain ranges ",
    r"^hills of ",
    r"^valleys of ",
    r"^plains of ",
    r"^plateaus of ",
    r"^deserts of ",
    r"^forests of ",
    r"^peninsulas of ",
    r"^capes of ",
    r"^volcanoes of ",
    r"^glaciers of ",
    r"^caves of ",
    r"^landforms of ",

    # Islands
    r"^islands of ",
    r"^islands in ",
    r"^archipelagoes of ",
    r"^atolls of ",

    # Populated places
    r"populated places in ",
    r"populated places of ",
    r"populated places established",
    r"former populated places",
    r"ghost towns",
    r"census-designated places",
    r"unincorporated communities",

    # Geography and location markers
    r"geography of ",
    r"landforms",
    r"coordinates ",
    r"places with ",
    r"locations in ",

    # Specific location types
    r"^ports in ",
    r"^port cities",
    r"^beach resorts",
    r"^ski resorts",
    r"^national parks",
    r"^protected areas",
    r"^world heritage sites",
    r"^tourist attractions in ",
]

# =============================================================================
# ORGANIZATION patterns - Companies, institutions, groups
# =============================================================================

ORGANIZATION_PATTERNS = [
    # Companies and businesses
    r"companies ",
    r"^companies based in ",
    r"^companies established in ",
    r"^companies of ",
    r"corporations",
    r"enterprises",
    r"conglomerates",
    r"multinationals",
    r"^brands ",
    r"^privately held companies",
    r"^publicly traded companies",
    r"^defunct companies",
    r"^holding companies",
    r"^subsidiaries",

    # Technology companies
    r"software companies",
    r"technology companies",
    r"computer companies",
    r"video game companies",
    r"internet companies",
    r"telecommunications companies",
    r"electronics companies",
    r"semiconductor companies",

    # Financial institutions
    r"^banks ",
    r"^banks of ",
    r"^investment banks",
    r"^central banks",
    r"insurance companies",
    r"financial services companies",
    r"hedge funds",
    r"private equity firms",
    r"venture capital firms",

    # Media and entertainment
    r"^newspapers ",
    r"^magazines ",
    r"^publishers ",
    r"^publishing companies",
    r"^television networks",
    r"^television stations",
    r"^radio stations",
    r"^film studios",
    r"^record labels",
    r"^media companies",
    r"^news agencies",

    # Educational institutions
    r"universities in ",
    r"universities and colleges in ",
    r"^universities of ",
    r"colleges in ",
    r"^schools in ",
    r"^high schools in ",
    r"^primary schools",
    r"^secondary schools",
    r"^boarding schools",
    r"^academies in ",
    r"^institutes of ",
    r"^research institutes",
    r"^libraries in ",
    r"^business schools",
    r"^law schools",
    r"^medical schools",

    # Healthcare
    r"^hospitals in ",
    r"^hospitals of ",
    r"^medical centers",
    r"^clinics in ",
    r"^pharmaceutical companies",
    r"^biotechnology companies",

    # Government and international
    r"^organizations ",
    r"^international organizations",
    r"^intergovernmental organizations",
    r"^non-governmental organizations",
    r"^nonprofit organizations",
    r"government agencies",
    r"^ministries of ",
    r"^departments of ",
    r"^united nations ",
    r"^european union ",

    # Political organizations
    r"political parties",
    r"^political parties in ",
    r"^conservative parties",
    r"^liberal parties",
    r"^socialist parties",
    r"^communist parties",
    r"^green parties",
    r"^nationalist parties",
    r"^political movements",
    r"^advocacy groups",
    r"^lobbying organizations",

    # Military and security
    r"^military units",
    r"^armies of ",
    r"^navies of ",
    r"^air forces of ",
    r"^intelligence agencies",
    r"^law enforcement agencies",
    r"^police departments",

    # Sports organizations
    r"^sports teams",
    r"^football teams",
    r"^football clubs",
    r"^soccer clubs",
    r"^basketball teams",
    r"^baseball teams",
    r"^hockey teams",
    r"^rugby clubs",
    r"^cricket teams",
    r"^sports leagues",
    r"^sports federations",
    r"^olympic teams",

    # Religious organizations
    r"^religious organizations",
    r"^churches in ",
    r"^mosques in ",
    r"^temples in ",
    r"^synagogues in ",
    r"^monasteries in ",
    r"^dioceses ",
    r"^religious orders",

    # Cultural organizations
    r"^museums in ",
    r"^art museums",
    r"^orchestras ",
    r"^theaters in ",
    r"^opera houses",
    r"^cultural institutions",
    r"^foundations ",
    r"^charities ",
    r"^think tanks",

    # Trade and professional
    r"trade unions",
    r"labor unions",
    r"professional associations",
    r"^industry associations",
    r"^chambers of commerce",

    # Other organizations
    r"^airlines ",
    r"^automotive companies",
    r"^retail companies",
    r"^restaurant chains",
    r"^hotel chains",
    r"^energy companies",
    r"^oil companies",
    r"^mining companies",
    r"^construction companies",
    r"^real estate companies",
    r"^transportation companies",
    r"^shipping companies",
    r"^aerospace companies",
    r"^defense companies",
]


@dataclass
class ClassificationResult:
    """Result of entity classification with confidence."""

    label: str
    confidence: float  # 0.0 to 1.0
    matched_patterns: List[str]


def classify_by_category(
    categories: List[str],
    return_confidence: bool = False
) -> Optional[str] | ClassificationResult:
    """Classify entity type based on Wikipedia categories.

    Args:
        categories: List of category names from the Wikipedia page.
        return_confidence: If True, return ClassificationResult with confidence.

    Returns:
        If return_confidence is False: "PER", "LOC", "ORG", or None if no match.
        If return_confidence is True: ClassificationResult object.
    """
    if not categories:
        if return_confidence:
            return ClassificationResult("MISC", 0.0, [])
        return None

    categories_lower = [c.lower() for c in categories]

    # Count matches for each type
    matches = {
        "PER": [],
        "LOC": [],
        "ORG": [],
    }

    # Check PERSON patterns
    for pattern in PERSON_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                matches["PER"].append(pattern)
                break

    # Check LOCATION patterns
    for pattern in LOCATION_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                matches["LOC"].append(pattern)
                break

    # Check ORGANIZATION patterns
    for pattern in ORGANIZATION_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                matches["ORG"].append(pattern)
                break

    # Determine winner
    match_counts = {k: len(v) for k, v in matches.items()}
    max_matches = max(match_counts.values())

    if max_matches == 0:
        if return_confidence:
            return ClassificationResult("MISC", 0.0, [])
        return None

    # Find the type with most matches
    winner = max(match_counts, key=match_counts.get)

    # Calculate confidence based on:
    # 1. Number of matching patterns
    # 2. Margin over second place
    second_place = sorted(match_counts.values(), reverse=True)[1] if len(match_counts) > 1 else 0
    margin = max_matches - second_place

    # Confidence formula: base confidence from matches + bonus for margin
    base_confidence = min(0.5 + (max_matches * 0.1), 0.9)
    margin_bonus = min(margin * 0.05, 0.1)
    confidence = min(base_confidence + margin_bonus, 1.0)

    if return_confidence:
        return ClassificationResult(winner, confidence, matches[winner])

    return winner


def get_category_signals(categories: List[str]) -> dict:
    """Analyze categories and return classification signals.

    Useful for debugging and understanding classification decisions.

    Args:
        categories: List of category names.

    Returns:
        Dict with match counts and matched patterns for each type.
    """
    if not categories:
        return {"PER": [], "LOC": [], "ORG": [], "total_categories": 0}

    categories_lower = [c.lower() for c in categories]

    signals = {
        "PER": [],
        "LOC": [],
        "ORG": [],
        "total_categories": len(categories),
    }

    for pattern in PERSON_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                signals["PER"].append({"pattern": pattern, "category": cat})
                break

    for pattern in LOCATION_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                signals["LOC"].append({"pattern": pattern, "category": cat})
                break

    for pattern in ORGANIZATION_PATTERNS:
        for cat in categories_lower:
            if re.search(pattern, cat, re.IGNORECASE):
                signals["ORG"].append({"pattern": pattern, "category": cat})
                break

    return signals
