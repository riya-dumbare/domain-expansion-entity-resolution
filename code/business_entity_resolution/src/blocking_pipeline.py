"""
Scalable blocking pipeline for Business Entity Resolution.

Dataset scale:
- Source 1: ~2.2M records
- Source 2: ~5.0M records
- Source 3: ~5.3M records

Design:
1. Partition by country to reduce the search space and memory usage.
2. Normalize names and addresses using vectorized pandas operations.
3. Generate candidates using token-based blocking with merge operations.
4. Remove overly frequent tokens to prevent large candidate blocks.
5. Process one country at a time to keep memory usage bounded.

Candidate recall is evaluated against the provided training ground truth.
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


def block_one_field(s1_tokens: pd.DataFrame, other_tokens: pd.DataFrame, other_id_col: str) -> pd.DataFrame:
    """Merge on shared token -> candidate (s1_id, other_id) pairs."""
    merged = s1_tokens.merge(other_tokens, on="token", suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", other_id_col]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "candidate_entity_id"]
    return pairs


def process_country(s1_c, s2_c, s3_c, max_token_freq=250):
    """Run blocking for a single country partition. Returns candidate pairs df."""
    s1_c = s1_c.assign(
        norm_name=normalize_name_series(s1_c["business_name"]),
        norm_addr=normalize_address_series(s1_c["business_address"]),
    )

    s1_c["name_initial"] = s1_c["norm_name"].str[:1]
    all_candidates = []

    for other_df in (s2_c, s3_c):
        if other_df.empty:
            continue

        other_df = other_df.assign(
            norm_name=normalize_name_series(other_df["business_name"]),
            norm_addr=normalize_address_series(other_df["business_address"]),
        )

        other_df["name_initial"] = other_df["norm_name"].str[:1]

        for initial in s1_c["name_initial"].dropna().unique():
            if initial == "":
                continue

            s1_part = s1_c[s1_c["name_initial"] == initial]
            other_part = other_df[other_df["name_initial"] == initial]

            if s1_part.empty or other_part.empty:
                continue

            s1_name_tok = purge_frequent_tokens(
                explode_tokens(s1_part, "entity_id", "norm_name"),
                max_token_freq,
            )

            other_name_tok = purge_frequent_tokens(
                explode_tokens(other_part, "entity_id", "norm_name"),
                max_token_freq,
            )

            name_pairs = block_one_field(
                s1_name_tok.rename(columns={"entity_id": "entity_id_s1"}),
                other_name_tok.rename(columns={"entity_id": "entity_id_other"}),
                "entity_id_other",
            )

            s1_addr_tok = purge_frequent_tokens(
                explode_tokens(s1_part, "entity_id", "norm_addr"),
                max_token_freq,
            )

            other_addr_tok = purge_frequent_tokens(
                explode_tokens(other_part, "entity_id", "norm_addr"),
                max_token_freq,
            )

            addr_pairs = block_one_field(
                s1_addr_tok.rename(columns={"entity_id": "entity_id_s1"}),
                other_addr_tok.rename(columns={"entity_id": "entity_id_other"}),
                "entity_id_other",
            )

            if not name_pairs.empty or not addr_pairs.empty:
                all_candidates.append(
                    pd.concat([name_pairs, addr_pairs]).drop_duplicates()
                )

    if not all_candidates:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )

    return pd.concat(all_candidates).drop_duplicates()

def evaluate_recall(candidates: pd.DataFrame, ground_truth_path: Path):
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    cand_sets = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    total_true, found_true = 0, 0
    for _, row in gt.iterrows():
        true_matches = [m for m in row["matched_entity_ids"].split(",") if m]
        cset = cand_sets.get(row["source1_entity_id"], set())
        for m in true_matches:
            total_true += 1
            if m in cset:
                found_true += 1

    recall_ceiling = found_true / total_true if total_true else 1.0
    print(f"\nRecall ceiling: {recall_ceiling:.4f}  ({found_true}/{total_true} true links recovered)")
    print(f"Total candidate pairs generated: {len(candidates):,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", required=True, type=Path)
    p.add_argument("--source2", required=True, type=Path)
    p.add_argument("--source3", required=True, type=Path)
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("output/candidate_pairs.tsv"))
    p.add_argument("--max-token-freq", type=int, default=250,
                    help="Drop blocking tokens more frequent than this (controls block size/memory)")
    args = p.parse_args()

    s1 = pd.read_csv(args.source1, sep="\t", dtype=str)
    s2 = pd.read_csv(args.source2, sep="\t", dtype=str)
    s3 = pd.read_csv(args.source3, sep="\t", dtype=str)

    countries = s1["country"].unique()
    print(f"Countries in source1: {list(countries)}")

    results = []
    for country in countries:
        print(f"\nProcessing country: {country}")
        s1_c = s1[s1["country"] == country]
        s2_c = s2[s2["country"] == country]
        s3_c = s3[s3["country"] == country]
        print(f"  s1={len(s1_c)} s2={len(s2_c)} s3={len(s3_c)}")
        pairs = process_country(s1_c, s2_c, s3_c, max_token_freq=args.max_token_freq)
        results.append(pairs)

    all_pairs = pd.concat(results).drop_duplicates()

    # Write in required format: one row per source1 entity
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grouped = all_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    grouped = grouped.reindex(s1["entity_id"]).fillna("")
    grouped.to_csv(args.out, sep="\t", header=["candidate_entity_ids"], index_label="source1_entity_id")
    print(f"\nWrote {len(grouped)} rows to {args.out}")

    if args.ground_truth:
        evaluate_recall(all_pairs, args.ground_truth)


if __name__ == "__main__":
    main()