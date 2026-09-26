"""Text preprocessing, normalization, and tokenization utilities for Entity Resolution.

Handles:
- Spelling errors, typos, and punctuation differences
- Business corporate suffixes and common abbreviations
- Address street/unit/directional abbreviations
- State abbreviations and country alias normalization
- Numeric component extraction (street numbers, zip codes, unit numbers)
- Safe handling of missing / empty / null values
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# Corporate designator canonicalization
CORP_ABBREVIATIONS: Dict[str, str] = {
    "incorporated": "inc",
    "corporation": "corp",
    "limited": "ltd",
    "company": "co",
    "companies": "co",
    "association": "assoc",
    "university": "univ",
    "department": "dept",
    "international": "intl",
    "management": "mgmt",
    "services": "svcs",
    "service": "svcs",
    "group": "grp",
    "holdings": "hldg",
    "holding": "hldg",
    "private": "pvt",
    "technologies": "tech",
    "technology": "tech",
    "solutions": "soln",
    "enterprises": "ent",
    "enterprise": "ent",
    "center": "ctr",
    "centre": "ctr",
    "brothers": "bros",
}

# Corporate suffixes to optionally strip for root name comparison
CORP_SUFFIXES: Set[str] = {
    "inc", "corp", "llc", "ltd", "co", "assoc", "pvt", "pllc", "pc",
    "sa", "gmbh", "bv", "ag", "sl", "srl", "spa", "lp", "llp"
}

# Address designators canonicalization
ADDRESS_ABBREVIATIONS: Dict[str, str] = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "parkway": "pkwy",
    "highway": "hwy",
    "freeway": "fwy",
    "square": "sq",
    "terrace": "ter",
    "trail": "trl",
    "way": "way",
    "suite": "ste",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "room": "rm",
    "department": "dept",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}

# Country name canonicalization
COUNTRY_ALIASES: Dict[str, str] = {
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "united kingdom": "gb",
    "great britain": "gb",
    "uk": "gb",
    "u.k.": "gb",
    "canada": "ca",
    "india": "in",
    "germany": "de",
    "deutschland": "de",
    "france": "fr",
    "australia": "au",
    "china": "cn",
    "japan": "jp",
    "brazil": "br",
    "mexico": "mx",
}

# US State full names to 2-letter postal code
US_STATE_CODES: Dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}

_PUNCT_REGEX = re.compile(r"[^\w\s]")
_MULTI_SPACE_REGEX = re.compile(r"\s+")
_DIGITS_REGEX = re.compile(r"\b\d+\b")


def safe_str(val: Optional[str]) -> str:
    """Convert None / NaN / missing values to clean string."""
    if val is None:
        return ""
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def basic_clean(text: Optional[str]) -> str:
    """Basic cleaning: lowercase, replace & with 'and', remove punctuation, normalize spaces."""
    s = safe_str(text).lower()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = _PUNCT_REGEX.sub(" ", s)
    s = _MULTI_SPACE_REGEX.sub(" ", s).strip()
    return s


def normalize_business_name(name: Optional[str]) -> Tuple[str, str]:
    """Clean and normalize business name.
    
    Returns:
        (canonical_name, root_name_without_corp_suffix)
    """
    cleaned = basic_clean(name)
    if not cleaned:
        return "", ""
    
    tokens = cleaned.split()
    normalized_tokens = [CORP_ABBREVIATIONS.get(t, t) for t in tokens]
    canonical_name = " ".join(normalized_tokens)
    
    # Root name: strip trailing corporate suffixes
    root_tokens = list(normalized_tokens)
    while root_tokens and root_tokens[-1] in CORP_SUFFIXES:
        root_tokens.pop()
    
    root_name = " ".join(root_tokens) if root_tokens else canonical_name
    return canonical_name, root_name


def normalize_address(address: Optional[str]) -> str:
    """Clean and normalize address components and abbreviations."""
    cleaned = basic_clean(address)
    if not cleaned:
        return ""
    
    tokens = cleaned.split()
    normalized_tokens = []
    i = 0
    while i < len(tokens):
        # Check two-word US state name (e.g., "north carolina" -> "nc")
        if i + 1 < len(tokens):
            two_word = f"{tokens[i]} {tokens[i+1]}"
            if two_word in US_STATE_CODES:
                normalized_tokens.append(US_STATE_CODES[two_word])
                i += 2
                continue
        t = tokens[i]
        t = US_STATE_CODES.get(t, t)
        t = ADDRESS_ABBREVIATIONS.get(t, t)
        normalized_tokens.append(t)
        i += 1
        
    return " ".join(normalized_tokens)


def extract_numeric_components(text: Optional[str]) -> List[str]:
    """Extract standalone numeric sequences (street numbers, zip codes, unit numbers)."""
    s = safe_str(text)
    if not s:
        return []
    return _DIGITS_REGEX.findall(s)


def normalize_country(country: Optional[str]) -> str:
    """Normalize country code/name."""
    cleaned = basic_clean(country)
    if not cleaned:
        return ""
    return COUNTRY_ALIASES.get(cleaned, cleaned)


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from text."""
    if not text:
        return set()
    s = f" {text} "
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def get_word_tokens(text: str) -> Set[str]:
    """Generate set of word tokens."""
    if not text:
        return set()
    return set(text.split())
