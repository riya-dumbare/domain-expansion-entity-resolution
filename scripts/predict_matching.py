#!/usr/bin/env python3

"""Memory-efficient test prediction for Business Entity Resolution."""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    DELIM,
    load_source_tsv,
    resolve_file_path,
)

from src.matching import (
    format_and_verify_results,
    score_candidates_batch,
    write_matching_results,
)

from src.matching_model import MatchingClassifier


def read_candidate_chunks(
    candidate_file: str,
    rows_per_chunk: int = 10000,
):
    """
    Read candidate_pairs.tsv in small chunks.

    Expected columns:
        source1_entity_id
        candidate_entity_ids

    candidate_entity_ids are expected to be comma-separated.
    """

    reader = pd.read_csv(
        candidate_file,
        sep=DELIM,
        dtype=str,
        chunksize=rows_per_chunk,
        keep_default_na=False,
    )

    for chunk_number, df in enumerate(reader, start=1):

        required = {
            "source1_entity_id",
            "candidate_entity_ids",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Candidate file must contain columns: {required}. "
                f"Found: {list(df.columns)}"
            )

        pairs: List[Tuple[str, str]] = []

        for _, row in df.iterrows():

            s1_id = str(row["source1_entity_id"]).strip()
            candidate_string = str(
                row["candidate_entity_ids"]
            ).strip()

            if not s1_id or not candidate_string:
                continue

            candidate_ids = candidate_string.split(",")

            for cand_id in candidate_ids:

                cand_id = cand_id.strip()

                if cand_id:
                    pairs.append(
                        (s1_id, cand_id)
                    )

        yield chunk_number, pairs


def main():

    parser = argparse.ArgumentParser(
        description="Memory-efficient prediction on test candidate pairs."
    )

    parser.add_argument(
        "--candidate-pairs",
        default="output/test_candidate_pairs.tsv",
    )

    parser.add_argument(
        "--test-source1",
        default="test_source1.tsv",
    )

    parser.add_argument(
        "--test-source2",
        default="test_source2.tsv",
    )

    parser.add_argument(
        "--test-source3",
        default="test_source3.tsv",
    )

    parser.add_argument(
        "--model",
        default="output/matching_model.joblib",
    )

    parser.add_argument(
        "--threshold-config",
        default="output/best_threshold.json",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=5000,
        help="Number of candidate rows read from TSV at once.",
    )

    parser.add_argument(
        "--output",
        default="output/matching_results.tsv",
    )

    parser.add_argument(
        "--skip-validator",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("MEMORY-SAFE TEST PREDICTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: Candidate file
    # ---------------------------------------------------------

    print("\n[Step 1/5] Checking candidate file...")

    cand_path = resolve_file_path(
        args.candidate_pairs,
        search_dirs=["output", "dataset/test", "."],
    )

    print(f"Candidate file: {cand_path}")

    # ---------------------------------------------------------
    # STEP 2: Load model
    # ---------------------------------------------------------

    print("\n[Step 2/5] Loading trained model...")

    model_path = resolve_file_path(
        args.model,
        search_dirs=["output", "models", "."],
    )

    model, saved_threshold = MatchingClassifier.load(
        model_path
    )

    print(
        f"Loaded model: {model.model_type}"
    )

    # ---------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------

    if args.threshold is not None:

        threshold = args.threshold

        print(
            f"Using manually specified threshold: "
            f"{threshold:.4f}"
        )

    elif os.path.isfile(args.threshold_config):

        with open(
            args.threshold_config,
            "r",
            encoding="utf-8",
        ) as f:

            config = json.load(f)

        threshold = float(
            config.get(
                "selected_threshold",
                saved_threshold or 0.5,
            )
        )

        print(
            f"Using threshold from config: "
            f"{threshold:.4f}"
        )

    else:

        threshold = saved_threshold or 0.5

        print(
            f"Using saved model threshold: "
            f"{threshold:.4f}"
        )

    # ---------------------------------------------------------
    # STEP 3: Load test entities
    # ---------------------------------------------------------

    print("\n[Step 3/5] Loading test entities...")

    s1_path = resolve_file_path(
        args.test_source1,
        search_dirs=["dataset/test", "."],
    )

    s1_entities = load_source_tsv(
        s1_path
    )

    all_s1_ids = set(
        s1_entities.keys()
    )

    print(
        f"Test Source 1: "
        f"{len(s1_entities):,}"
    )

    target_entities: Dict[
        str,
        Tuple[
            Optional[str],
            Optional[str],
            Optional[str],
        ],
    ] = {}

    # Source 2

    s2_path = resolve_file_path(
        args.test_source2,
        search_dirs=["dataset/test", "."],
        must_exist=False,
    )

    if os.path.isfile(s2_path):

        s2_entities = load_source_tsv(
            s2_path
        )

        target_entities.update(
            s2_entities
        )

        print(
            f"Test Source 2: "
            f"{len(s2_entities):,}"
        )

    # Source 3

    s3_path = resolve_file_path(
        args.test_source3,
        search_dirs=["dataset/test", "."],
    )

    s3_entities = load_source_tsv(
        s3_path
    )

    target_entities.update(
        s3_entities
    )

    print(
        f"Test Source 3: "
        f"{len(s3_entities):,}"
    )

    print(
        f"Total target entities: "
        f"{len(target_entities):,}"
    )

    # ---------------------------------------------------------
    # STEP 4: Chunk-by-chunk prediction
    # ---------------------------------------------------------

    print(
        "\n[Step 4/5] "
        "Starting memory-safe candidate scoring..."
    )

    print(
        f"Candidate chunk size: "
        f"{args.candidate_chunk_size:,} rows"
    )

    print(
        f"Model batch size: "
        f"{args.batch_size:,} pairs"
    )

    matches_by_s1: Dict[
        str,
        List[Tuple[str, float]]
    ] = {}

    total_pairs = 0
    total_passed = 0
    total_chunks = 0

    for chunk_number, candidate_pairs in read_candidate_chunks(
        cand_path,
        rows_per_chunk=args.candidate_chunk_size,
    ):

        total_chunks += 1

        if not candidate_pairs:
            continue

        print(
            f"\n--- Candidate chunk "
            f"{chunk_number} ---"
        )

        print(
            f"Pairs in chunk: "
            f"{len(candidate_pairs):,}"
        )

        total_pairs += len(candidate_pairs)

        chunk_matches, chunk_passed = score_candidates_batch(
            candidate_pairs=candidate_pairs,
            source1_entities=s1_entities,
            target_entities=target_entities,
            model=model,
            threshold=threshold,
            batch_size=args.batch_size,
            verbose=False,
        )

        for s1_id, matches in chunk_matches.items():

            if s1_id not in matches_by_s1:

                matches_by_s1[s1_id] = []

            matches_by_s1[s1_id].extend(
                matches
            )

        total_passed += chunk_passed

        print(
            f"Chunk matches: "
            f"{chunk_passed:,}"
        )

        print(
            f"Total pairs processed: "
            f"{total_pairs:,}"
        )

        print(
            f"Total matches so far: "
            f"{total_passed:,}"
        )

    print("\nCandidate scoring completed.")

    print(
        f"Total chunks: "
        f"{total_chunks:,}"
    )

    print(
        f"Total candidate pairs processed: "
        f"{total_pairs:,}"
    )

    print(
        f"Total pairs passing threshold: "
        f"{total_passed:,}"
    )

    # ---------------------------------------------------------
    # STEP 5: Format final results
    # ---------------------------------------------------------

    print(
        "\n[Step 5/5] "
        "Formatting final matching_results.tsv..."
    )

    results = format_and_verify_results(
        all_source1_ids=all_s1_ids,
        matches_by_s1=matches_by_s1,
        candidate_map=None,
    )

    out_file = write_matching_results(
        results,
        args.output,
    )

    non_empty = sum(
        1
        for _, matched in results
        if matched.strip()
    )

    empty_count = (
        len(results) - non_empty
    )

    print(
        f"\nSuccessfully generated:"
    )

    print(out_file)

    print(
        f"Total Source 1 entities: "
        f"{len(results):,}"
    )

    print(
        f"Entities with matches: "
        f"{non_empty:,}"
    )

    print(
        f"Entities without match: "
        f"{empty_count:,}"
    )

    # ---------------------------------------------------------
    # Official validator
    # ---------------------------------------------------------

    if not args.skip_validator:

        validator_script = resolve_file_path(
            "validate_submission.py",
            search_dirs=[".", "utils"],
            must_exist=False,
        )

        if os.path.isfile(
            validator_script
        ):

            print(
                "\nRunning official "
                "submission validator..."
            )

            cmd = [
                sys.executable,
                validator_script,
                "--matching",
                out_file,
                "--candidate",
                cand_path,
                "--test-dir",
                os.path.dirname(
                    s1_path
                ) or ".",
            ]

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            print(res.stdout)

            if res.stderr:
                print(res.stderr)

            if res.returncode == 0:

                print(
                    ">>> OFFICIAL VALIDATOR PASSED <<<"
                )

            else:

                print(
                    f">>> Validator exited with "
                    f"code {res.returncode} <<<"
                )

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION COMPLETED"
    )

    print(
        f"Final output: {out_file}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()