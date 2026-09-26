"""Feature Engineering module for Entity Resolution candidate pairs.

Computes similarity features across business name, address, numeric components, and country:
- Business name edit similarity (Levenshtein)
- Business name token similarity (token sort, token set)
- Business name Jaccard similarity (word tokens & char 3-grams)
- Address edit similarity
- Address token similarity
- Address Jaccard similarity
- Address numeric/component similarity
- Country agreement
- Missingness indicator features
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.text_processing import (
    extract_numeric_components,
    get_character_ngrams,
    get_word_tokens,
    normalize_address,
    normalize_business_name,
    normalize_country,
    safe_str,
)

# Try importing rapidfuzz for blazing fast similarity calculations
try:
    from rapidfuzz.distance import Levenshtein as rf_lev
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


FEATURE_NAMES = [
    # Business name features
    "name_edit_sim",
    "name_root_edit_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_jaccard_word",
    "name_jaccard_char_3gram",
    "name_exact_match",
    "name_length_diff",
    # Address features
    "addr_edit_sim",
    "addr_token_sort_sim",
    "addr_token_set_sim",
    "addr_jaccard_word",
    "addr_jaccard_char_3gram",
    "addr_exact_match",
    # Address numeric/component features
    "addr_numeric_exact",
    "addr_numeric_jaccard",
    "addr_numeric_has_overlap",
    # Country features
    "country_match",
    # Missingness indicator features
    "name_missing",
    "addr_missing",
    "country_missing",
]


def _levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return rf_lev.normalized_similarity(s1, s2)
    
    # Pure Python fallback
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    dp = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    return max(0.0, 1.0 - (dp[len2] / max_len))


def _token_sort_sim(s1: str, s2: str) -> float:
    """Token sort similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return rf_fuzz.token_sort_ratio(s1, s2) / 100.0
    
    tokens1 = sorted(s1.split())
    tokens2 = sorted(s2.split())
    return _levenshtein_sim(" ".join(tokens1), " ".join(tokens2))


def _token_set_sim(s1: str, s2: str) -> float:
    """Token set similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return rf_fuzz.token_set_ratio(s1, s2) / 100.0
    
    set1, set2 = set(s1.split()), set(s2.split())
    intersection = " ".join(sorted(set1 & set2))
    diff1 = " ".join(sorted(set1 - set2))
    diff2 = " ".join(sorted(set2 - set1))
    
    s_int1 = f"{intersection} {diff1}".strip()
    s_int2 = f"{intersection} {diff2}".strip()
    return max(_levenshtein_sim(intersection, s_int1),
               _levenshtein_sim(intersection, s_int2),
               _levenshtein_sim(s_int1, s_int2))


def _jaccard_sim(set1: Set[Any], set2: Set[Any]) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection_len = len(set1 & set2)
    union_len = len(set1 | set2)
    if union_len == 0:
        return 0.0
    return intersection_len / union_len


class ProcessedEntity:
    """Cached representation of an entity for fast pair feature calculation."""
    __slots__ = (
        "raw_name", "raw_address", "raw_country",
        "norm_name", "root_name", "name_tokens", "name_ngrams", "has_name",
        "norm_address", "addr_tokens", "addr_ngrams", "numeric_tokens", "has_address",
        "norm_country", "has_country"
    )

    def __init__(self, name: Optional[str], address: Optional[str], country: Optional[str]):
        self.raw_name = safe_str(name)
        self.raw_address = safe_str(address)
        self.raw_country = safe_str(country)

        # Name processing
        norm_name, root_name = normalize_business_name(self.raw_name)
        self.norm_name = norm_name
        self.root_name = root_name
        self.name_tokens = get_word_tokens(norm_name)
        self.name_ngrams = get_character_ngrams(norm_name, 3)
        self.has_name = bool(norm_name)

        # Address processing
        norm_addr = normalize_address(self.raw_address)
        self.norm_address = norm_addr
        self.addr_tokens = get_word_tokens(norm_addr)
        self.addr_ngrams = get_character_ngrams(norm_addr, 3)
        self.numeric_tokens = set(extract_numeric_components(self.raw_address))
        self.has_address = bool(norm_addr)

        # Country processing
        norm_ctry = normalize_country(self.raw_country)
        self.norm_country = norm_ctry
        self.has_country = bool(norm_ctry)


def compute_pair_features(e1: ProcessedEntity, e2: ProcessedEntity) -> List[float]:
    """Compute all 21 similarity and indicator features for a candidate pair."""
    # 1. Business name features
    if e1.has_name and e2.has_name:
        name_edit = _levenshtein_sim(e1.norm_name, e2.norm_name)
        name_root_edit = _levenshtein_sim(e1.root_name, e2.root_name)
        name_token_sort = _token_sort_sim(e1.norm_name, e2.norm_name)
        name_token_set = _token_set_sim(e1.norm_name, e2.norm_name)
        name_jaccard_w = _jaccard_sim(e1.name_tokens, e2.name_tokens)
        name_jaccard_ng = _jaccard_sim(e1.name_ngrams, e2.name_ngrams)
        name_exact = 1.0 if e1.norm_name == e2.norm_name else 0.0
        max_l = max(len(e1.norm_name), len(e2.norm_name))
        name_len_diff = abs(len(e1.norm_name) - len(e2.norm_name)) / max_l if max_l > 0 else 0.0
        name_missing = 0.0
    else:
        name_edit = 0.0
        name_root_edit = 0.0
        name_token_sort = 0.0
        name_token_set = 0.0
        name_jaccard_w = 0.0
        name_jaccard_ng = 0.0
        name_exact = 0.0
        name_len_diff = 1.0
        name_missing = 1.0

    # 2. Address features
    if e1.has_address and e2.has_address:
        addr_edit = _levenshtein_sim(e1.norm_address, e2.norm_address)
        addr_token_sort = _token_sort_sim(e1.norm_address, e2.norm_address)
        addr_token_set = _token_set_sim(e1.norm_address, e2.norm_address)
        addr_jaccard_w = _jaccard_sim(e1.addr_tokens, e2.addr_tokens)
        addr_jaccard_ng = _jaccard_sim(e1.addr_ngrams, e2.addr_ngrams)
        addr_exact = 1.0 if e1.norm_address == e2.norm_address else 0.0
        addr_missing = 0.0
    else:
        addr_edit = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard_w = 0.0
        addr_jaccard_ng = 0.0
        addr_exact = 0.0
        addr_missing = 1.0

    # 3. Numeric / component features
    nums1, nums2 = e1.numeric_tokens, e2.numeric_tokens
    if nums1 and nums2:
        addr_num_exact = 1.0 if nums1 == nums2 else 0.0
        addr_num_jaccard = _jaccard_sim(nums1, nums2)
        addr_num_overlap = 1.0 if bool(nums1 & nums2) else 0.0
    elif not nums1 and not nums2:
        # Neither has numbers: neutral
        addr_num_exact = 0.5
        addr_num_jaccard = 0.5
        addr_num_overlap = 0.5
    else:
        # One has numbers, the other doesn't
        addr_num_exact = 0.0
        addr_num_jaccard = 0.0
        addr_num_overlap = 0.0

    # 4. Country agreement feature
    if e1.has_country and e2.has_country:
        country_match = 1.0 if e1.norm_country == e2.norm_country else 0.0
        country_missing = 0.0
    else:
        country_match = 0.5  # Unknown/neutral
        country_missing = 1.0

    return [
        name_edit,
        name_root_edit,
        name_token_sort,
        name_token_set,
        name_jaccard_w,
        name_jaccard_ng,
        name_exact,
        name_len_diff,
        addr_edit,
        addr_token_sort,
        addr_token_set,
        addr_jaccard_w,
        addr_jaccard_ng,
        addr_exact,
        addr_num_exact,
        addr_num_jaccard,
        addr_num_overlap,
        country_match,
        name_missing,
        addr_missing,
        country_missing,
    ]


def build_entity_cache(records: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]) -> Dict[str, ProcessedEntity]:
    """Preprocess and cache entities into ProcessedEntity objects."""
    cache: Dict[str, ProcessedEntity] = {}
    for eid, (name, addr, ctry) in records.items():
        cache[eid] = ProcessedEntity(name, addr, ctry)
    return cache


def compute_features_matrix(
    candidate_pairs: List[Tuple[str, str]],
    source1_cache: Dict[str, ProcessedEntity],
    target_cache: Dict[str, ProcessedEntity],
) -> np.ndarray:
    """Compute feature matrix for a list of candidate pairs (source1_id, candidate_id)."""
    n_pairs = len(candidate_pairs)
    if n_pairs == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    X = np.zeros((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    dummy_entity = ProcessedEntity("", "", "")

    for i, (s1_id, cand_id) in enumerate(candidate_pairs):
        e1 = source1_cache.get(s1_id, dummy_entity)
        e2 = target_cache.get(cand_id, dummy_entity)
        X[i] = compute_pair_features(e1, e2)

    return X
