"""
Scalable blocking pipeline for Business Entity Resolution.

Rewritten for the ACTUAL dataset scale:
  source1 ~2.2M rows, source2 ~5.0M rows, source3 ~5.3M rows.

Key design decisions (each addresses a real constraint found via EDA):

1. COUNTRY PARTITIONING: ground truth showed 0 cross-country matches out of
   1.8M+ links checked. So we solve each country independently:
   candidates for a US Source1 entity only ever come from US Source2/3, etc.
   This roughly halves the search space per partition and lets you process
   one country at a time to control memory.

2. VECTORIZED NORMALIZATION: no per-row Python loops. Uses pandas .str
   methods (regex-backed, implemented in C) across the whole column at once.

3. TOKEN-EXPLODE + MERGE BLOCKING: instead of building a Python dict/set
   inverted index with iterrows() (infeasible at this scale), tokens are
   exploded into a long dataframe and joined via pandas merge (a hash join),
   which is what pandas is actually fast at.

4. FREQUENT-TOKEN PURGING: without this, common tokens (e.g. "street",
   "delhi", "llc") would create enormous candidate blocks and blow up both
   memory and precision. Tokens appearing in more than `max_token_freq`
   records on the Source2/3 side are dropped as blocking keys (they're too
   generic to discriminate between businesses anyway).

Memory note: loading all 3 files fully at once can exceed a few GB of RAM.
This script processes ONE COUNTRY AT A TIME to keep peak memory bounded.
If you still hit memory issues, reduce max_token_freq (smaller blocks) or
process source2 and source3 separately rather than concatenated.

Run:
    python src/blocking_pipeline.py \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --out output/candidate_pairs.tsv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:incorporated|inc|corporation|corp|limited|ltd|private|pvt|llc|llp|company|co|plc)\b",
    re.UNICODE,
)
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")

ADDRESS_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "apt": "apartment",
    "bldg": "building", "flr": "floor", "nr": "near",
}


def normalize_name_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    s = s.str.replace(LEGAL_SUFFIX_RE, " ", regex=True)
    s = s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()
    return s


def normalize_address_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    for abbr, full in ADDRESS_ABBR.items():
        s = s.str.replace(rf"\b{abbr}\b", full, regex=True)
    s = s.str.replace(MULTISPACE_RE, " ", regex=True).str.strip()
    return s


def explode_tokens(df: pd.DataFrame, id_col: str, text_col: str, min_len: int = 3) -> pd.DataFrame:
    """Long-format (entity_id, token) table, one row per token per record."""
    tmp = df[[id_col, text_col]].copy()
    tmp["token"] = tmp[text_col].str.split()
    tmp = tmp.explode("token").dropna(subset=["token"])
    tmp = tmp[tmp["token"].str.len() >= min_len]
    return tmp[[id_col, "token"]]


def purge_frequent_tokens(token_df: pd.DataFrame, max_freq: int) -> pd.DataFrame:
    """Drop tokens that appear in more than max_freq distinct entities.
    These are too generic to be useful blocking keys and are the main
    cause of blowout in join size."""
    freq = token_df["token"].value_counts()
    keep = freq[freq <= max_freq].index
    return token_df[token_df["token"].isin(keep)]


def process_country(s1_c, s2_c, s3_c, max_token_freq=25):
    """Run blocking for a single country partition.

    Returns a dict {source1_entity_id: set(candidate_entity_id)}.

    IMPORTANT: this aggregates to sets immediately after each merge, instead
    of concatenating every raw pair across both fields and both sources into
    one giant DataFrame before deduplicating. The earlier version did that
    final concat+drop_duplicates over millions of string pairs, which is what
    caused the MemoryError on the full dataset (1.3M x 3M+ records) even
    though each individual merge was purge-bounded. Aggregating per-step and
    discarding the raw merged frame keeps peak memory to one merge's worth
    at a time, not the whole country's worth.
    """
    s1_c = s1_c.assign(
        norm_name=normalize_name_series(s1_c["business_name"]),
        norm_addr=normalize_address_series(s1_c["business_address"]),
    )
    candidate_sets: dict[str, set] = {}

    for other_df in (s2_c, s3_c):
        if other_df.empty:
            continue
        other_df = other_df.assign(
            norm_name=normalize_name_series(other_df["business_name"]),
            norm_addr=normalize_address_series(other_df["business_address"]),
        )

        for text_col in ("norm_name", "norm_addr"):
            s1_tok = purge_frequent_tokens(explode_tokens(s1_c, "entity_id", text_col), max_token_freq)
            other_tok = purge_frequent_tokens(explode_tokens(other_df, "entity_id", text_col), max_token_freq)
            if s1_tok.empty or other_tok.empty:
                continue

            merged = s1_tok.merge(
                other_tok, on="token", suffixes=("_s1", "_other")
            )[["entity_id_s1", "entity_id_other"]]

            # Aggregate THIS merge's pairs into the running dict of sets right
            # away, then let `merged` be garbage-collected before the next
            # field/source iteration — this is what bounds peak memory.
            grouped = merged.groupby("entity_id_s1")["entity_id_other"].apply(set)
            for s1_id, others in grouped.items():
                if s1_id in candidate_sets:
                    candidate_sets[s1_id] |= others
                else:
                    candidate_sets[s1_id] = others
            del merged, grouped

    return candidate_sets


def evaluate_recall(candidate_sets: dict, ground_truth_path: Path):
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    total_true, found_true = 0, 0
    total_candidates = 0
    for s1_id, cset in candidate_sets.items():
        total_candidates += len(cset)

    for _, row in gt.iterrows():
        true_matches = [m for m in row["matched_entity_ids"].split(",") if m]
        cset = candidate_sets.get(row["source1_entity_id"], set())
        for m in true_matches:
            total_true += 1
            if m in cset:
                found_true += 1

    recall_ceiling = found_true / total_true if total_true else 1.0
    print(f"\nRecall ceiling: {recall_ceiling:.4f}  ({found_true}/{total_true} true links recovered)")
    print(f"Total candidate pairs generated: {total_candidates:,}")
    n_entities = len(candidate_sets) if candidate_sets else 1
    print(f"Avg candidates per Source1 entity (with any candidates): {total_candidates / n_entities:.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", required=True, type=Path)
    p.add_argument("--source2", required=True, type=Path)
    p.add_argument("--source3", required=True, type=Path)
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("output/candidate_pairs.tsv"))
    p.add_argument("--max-token-freq", type=int, default=100,
                    help="Drop blocking tokens more frequent than this (controls block size/memory). "
                         "Lower = tighter blocking (less memory, lower recall ceiling), higher = looser "
                         "(more memory/candidates, higher recall). This is scale-dependent: the right "
                         "value differs between country partitions and between train/test sizes, so "
                         "re-tune it by watching the printed recall ceiling and avg-candidates-per-entity "
                         "whenever you run on a meaningfully different data size.")
    args = p.parse_args()

    s1 = pd.read_csv(args.source1, sep="\t", dtype=str)
    s2 = pd.read_csv(args.source2, sep="\t", dtype=str)
    s3 = pd.read_csv(args.source3, sep="\t", dtype=str)

    countries = s1["country"].unique()
    print(f"Countries in source1: {list(countries)}")

    all_candidate_sets: dict = {}
    for country in countries:
        print(f"\nProcessing country: {country}")
        s1_c = s1[s1["country"] == country]
        s2_c = s2[s2["country"] == country]
        s3_c = s3[s3["country"] == country]
        print(f"  s1={len(s1_c)} s2={len(s2_c)} s3={len(s3_c)}")
        country_sets = process_country(s1_c, s2_c, s3_c, max_token_freq=args.max_token_freq)
        all_candidate_sets.update(country_sets)
        n_cand = sum(len(v) for v in country_sets.values())
        print(f"  -> {len(country_sets)} entities with candidates, {n_cand:,} total candidate pairs")

    # Write in required format: one row per source1 entity, in source1's own order
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1["entity_id"]:
            ids = sorted(all_candidate_sets.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(ids)}\n")
    print(f"\nWrote {len(s1)} rows to {args.out}")

    if args.ground_truth:
        evaluate_recall(all_candidate_sets, args.ground_truth)


if __name__ == "__main__":
    main()