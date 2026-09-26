#!/usr/bin/env python3
"""Generate a complete, submission-compliant mock output/candidate_pairs.tsv for temporary testing.

Why this script exists:
Member 1 is responsible for generating 'output/candidate_pairs.tsv' (blocking).
While Member 1 is completing their work, this script generates a temporary mock
'output/candidate_pairs.tsv' that:
1. Matches sample entities from test_source1.tsv against test_source3.tsv.
2. Includes ALL 1,732,544 entities from test_source1.tsv (with candidates for sample entities
   and empty rows for the rest).
3. Conforms 100% to the official challenge schema so validate_submission.py passes with Exit Code 0.

Once Member 1 uploads the real 'output/candidate_pairs.tsv', it simply overwrites this mock.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import DELIM, load_source_tsv, resolve_file_path
from src.text_processing import basic_clean


def main():
    parser = argparse.ArgumentParser(description="Generate complete mock candidate_pairs.tsv for temporary testing")
    parser.add_argument("--test-s1", default="test_source1.tsv", help="Test Source 1 TSV path")
    parser.add_argument("--test-s3", default="test_source3.tsv", help="Test Source 3 TSV path")
    parser.add_argument("--test-s2", default="test_source2.tsv", help="Test Source 2 TSV path (optional)")
    parser.add_argument("--output", default="output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--num-candidates", type=int, default=5000, help="Number of entities to generate candidate pairs for")
    args = parser.parse_args()

    # Reconfigure stdout for UTF-8 display
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("MEMBER 1 MOCK GENERATOR: TEMPORARY candidate_pairs.tsv")
    print("=" * 70)

    # 1. Resolve paths
    s1_path = resolve_file_path(args.test_s1, search_dirs=["dataset/test", "."])
    s3_path = resolve_file_path(args.test_s3, search_dirs=["dataset/test", "."])

    print(f"Reading target index from {s3_path}...")
    # Load a slice of target records to form a blocking index
    target_dict = load_source_tsv(s3_path, max_rows=150000)

    # Also check test_source2 if present
    try:
        s2_path = resolve_file_path(args.test_s2, search_dirs=["dataset/test", "."], must_exist=False)
        if os.path.isfile(s2_path):
            s2_dict = load_source_tsv(s2_path, max_rows=100000)
            target_dict.update(s2_dict)
            print(f"Loaded additional targets from {s2_path}")
    except Exception:
        pass

    print(f"Indexing {len(target_dict):,} target entities by first token...")
    target_index = {}
    for eid, (name, _, _) in target_dict.items():
        cname = basic_clean(name)
        if cname:
            tok = cname.split()[0]
            if tok not in target_index:
                target_index[tok] = []
            if len(target_index[tok]) < 10:
                target_index[tok].append(eid)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_path = os.path.abspath(args.output)
    print(f"Streaming through {s1_path} to create complete candidate_pairs.tsv...")

    total_s1 = 0
    with_cands = 0
    total_pairs = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        
        # Write exact required header
        fout.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
        
        header = fin.readline()
        header_cols = [c.strip().lower() for c in header.split(DELIM)]
        id_idx = header_cols.index("entity_id") if "entity_id" in header_cols else 0
        name_idx = header_cols.index("business_name") if "business_name" in header_cols else 1

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(DELIM)
            if len(parts) <= id_idx:
                continue
            
            s1_id = parts[id_idx].strip()
            if not s1_id:
                continue
            
            total_s1 += 1
            candidates = []

            # Generate candidate matches for the first `num_candidates` entities
            if with_cands < args.num_candidates:
                s1_name = parts[name_idx].strip() if len(parts) > name_idx else ""
                cname = basic_clean(s1_name)
                if cname:
                    tok = cname.split()[0]
                    candidates = target_index.get(tok, [])
                    if candidates:
                        with_cands += 1
                        total_pairs += len(candidates)

            cand_str = ",".join(candidates)
            fout.write(f"{s1_id}{DELIM}{cand_str}\n")

            if total_s1 % 500000 == 0:
                print(f"  Processed {total_s1:,} entities...")

    print(f"\nDone! Successfully created: {out_path}")
    print(f"  Total Source 1 entities written: {total_s1:,} (100% of test_source1.tsv)")
    print(f"  Entities with candidate pairs:  {with_cands:,} ({total_pairs:,} total candidate pairs)")
    print(f"  Entities with empty candidate:  {total_s1 - with_cands:,}")
    print(f"\nThis temporary mock candidate file allows full end-to-end testing of Member 2!")


if __name__ == "__main__":
    main()
