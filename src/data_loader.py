"""Data loading utilities for TSV files, entity dictionaries, ground truth, and candidate pairs.

Provides:
- Tab-delimited TSV reading with UTF-8 encoding
- Flexible file path resolution (dataset/ directories or root directory)
- Schema inspection and validation
- Dynamic candidate_pairs.tsv format adaptation (list-format or pairwise)
- Ground truth parsing into set-based lookups
- Robust error checking for missing files, bad columns, and format errors
"""

import os
from typing import Dict, List, Optional, Set, Tuple


DELIM = "\t"


def resolve_file_path(
    filename: str,
    search_dirs: Optional[List[str]] = None,
    must_exist: bool = True
) -> str:
    """Find a file by checking specified directories or standard locations.
    
    Checks in order:
    1. Direct path as given
    2. Under search_dirs
    3. Under dataset/train or dataset/test or output/ or current working directory
    """
    if os.path.isfile(filename):
        return os.path.abspath(filename)
    
    candidates = []
    if search_dirs:
        for d in search_dirs:
            candidates.append(os.path.join(d, filename))
    basename = os.path.basename(filename)
    standard_locations = [
        os.path.join(".", basename),
        os.path.join("models", basename),
        os.path.join("output", basename),
        os.path.join("dataset", "train", basename),
        os.path.join("dataset", "test", basename),
        os.path.join("dataset", basename),
    ]
    candidates.extend(standard_locations)

    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    if must_exist:
        raise FileNotFoundError(
            f"Could not locate '{filename}'. Checked: {[os.path.abspath(p) for p in candidates[:6]]}"
        )
    return filename


def load_source_tsv(
    file_path: str,
    max_rows: Optional[int] = None,
) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]:
    """Load an entity source TSV into a memory-efficient dict: {entity_id: (name, address, country)}.
    
    Expected columns: entity_id, business_name, business_address, country
    """
    resolved = resolve_file_path(file_path)
    entities: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Source file {resolved} is completely empty.")
        
        headers = [c.strip().lower() for c in header_line.rstrip("\n").split(DELIM)]
        
        # Validate columns
        if "entity_id" not in headers:
            raise ValueError(
                f"Source file {resolved} missing 'entity_id' column. Found headers: {headers}"
            )
        
        id_idx = headers.index("entity_id")
        name_idx = headers.index("business_name") if "business_name" in headers else -1
        addr_idx = headers.index("business_address") if "business_address" in headers else -1
        ctry_idx = headers.index("country") if "country" in headers else -1

        for line_num, line in enumerate(f, start=2):
            if max_rows and len(entities) >= max_rows:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            if id_idx >= len(parts):
                continue
            
            eid = parts[id_idx].strip()
            if not eid:
                continue
            
            name = parts[name_idx].strip() if (name_idx >= 0 and name_idx < len(parts)) else None
            addr = parts[addr_idx].strip() if (addr_idx >= 0 and addr_idx < len(parts)) else None
            ctry = parts[ctry_idx].strip() if (ctry_idx >= 0 and ctry_idx < len(parts)) else None

            entities[eid] = (name, addr, ctry)

    return entities


def load_ground_truth(
    file_path: str,
    max_rows: Optional[int] = None
) -> Dict[str, Set[str]]:
    """Load ground truth mappings: {source1_entity_id: {matched_s2_or_s3_ids}}."""
    resolved = resolve_file_path(file_path)
    gt_map: Dict[str, Set[str]] = {}

    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Ground truth file {resolved} is empty.")
        
        headers = [c.strip().lower() for c in header_line.rstrip("\n").split(DELIM)]
        
        s1_col = -1
        matches_col = -1
        for idx, h in enumerate(headers):
            if "source1" in h or h == "s1_id" or h == "entity_id":
                s1_col = idx
            elif "matched" in h or "match" in h or "target" in h:
                matches_col = idx
                
        if s1_col == -1 or matches_col == -1:
            # Fallback to standard 0 and 1 indices
            s1_col, matches_col = 0, 1

        for line_num, line in enumerate(f, start=2):
            if max_rows and len(gt_map) >= max_rows:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            if len(parts) <= s1_col:
                continue
            
            s1_id = parts[s1_col].strip()
            if not s1_id:
                continue
            
            raw_matches = parts[matches_col].strip() if len(parts) > matches_col else ""
            if raw_matches:
                # Comma separated target IDs
                matched_set = {mid.strip() for mid in raw_matches.split(",") if mid.strip()}
            else:
                matched_set = set()
                
            gt_map[s1_id] = matched_set

    return gt_map


def load_candidate_pairs(
    file_path: str,
    max_pairs: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, Set[str]], List[str]]:
    """Load candidate pairs, automatically inspecting and adapting to the file's schema.
    
    Supports both:
    1. Grouped list format (official validate_submission format):
       source1_entity_id \\t candidate_entity_ids (comma-separated S2/S3 IDs)
    2. Pairwise format:
       source1_entity_id \\t candidate_entity_id
       
    Returns:
        (flat_pair_list, s1_to_candidates_map, column_names_detected)
    """
    resolved = resolve_file_path(file_path)
    
    flat_pairs: List[Tuple[str, str]] = []
    s1_to_candidates: Dict[str, Set[str]] = {}
    
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Candidate file {resolved} is empty.")
        
        raw_headers = [c.strip() for c in header_line.rstrip("\n").split(DELIM)]
        headers_lower = [c.lower() for c in raw_headers]
        
        # Check column count
        if len(headers_lower) < 2:
            raise ValueError(
                f"Candidate file {resolved} has invalid format with < 2 tab-separated columns: {raw_headers}"
            )
        
        # Detect column roles
        s1_idx = 0
        cand_idx = 1
        for idx, col in enumerate(headers_lower):
            if "source1" in col or "s1" in col:
                s1_idx = idx
            elif "candidate" in col or "target" in col or "s2" in col or "s3" in col or "match" in col:
                cand_idx = idx
                
        is_list_format = "candidate_entity_ids" in headers_lower or "matched_entity_ids" in headers_lower
        
        for line_num, line in enumerate(f, start=2):
            if max_pairs and len(flat_pairs) >= max_pairs:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            if len(parts) <= s1_idx:
                continue
            
            s1_id = parts[s1_idx].strip()
            if not s1_id:
                continue
            
            cand_raw = parts[cand_idx].strip() if len(parts) > cand_idx else ""
            if s1_id not in s1_to_candidates:
                s1_to_candidates[s1_id] = set()

            if not cand_raw:
                continue
            
            # Check if comma-separated or single candidate
            if "," in cand_raw or is_list_format:
                cands = [c.strip() for c in cand_raw.split(",") if c.strip()]
            else:
                cands = [cand_raw]

            for c in cands:
                # Sanity check: candidate IDs must not be S1 IDs (no self-matches)
                if c.startswith("S1-"):
                    continue
                if c not in s1_to_candidates[s1_id]:
                    s1_to_candidates[s1_id].add(c)
                    flat_pairs.append((s1_id, c))
                    if max_pairs and len(flat_pairs) >= max_pairs:
                        break

    return flat_pairs, s1_to_candidates, raw_headers
