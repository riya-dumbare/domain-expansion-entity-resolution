import pandas as pd
from pathlib import Path


CANDIDATE_FILE = Path("output/candidate_pairs.tsv")
GROUND_TRUTH_FILE = Path("data/train_ground_truth.tsv")


def validate_candidate_file():

    if not CANDIDATE_FILE.exists():
        print("candidate_pairs.tsv not found.")
        return

    print("Candidate file found.")

    # Check columns
    header = pd.read_csv(CANDIDATE_FILE, sep="\t", nrows=0)

    required_columns = [
        "source1_entity_id",
        "candidate_entity_ids"
    ]

    print("Columns:", list(header.columns))

    if list(header.columns) != required_columns:
        print("ERROR: Wrong columns.")
        return

    print("PASS: Correct columns.")

    # Validate in chunks
    total_rows = 0
    duplicate_ids = 0

    seen_ids = set()

    for chunk in pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=100000
    ):
        chunk = chunk.fillna("")

        total_rows += len(chunk)

        for entity_id in chunk["source1_entity_id"]:
            if entity_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(entity_id)

        print("Checked rows:", total_rows)

    print("\n==============================")
    print("VALIDATION RESULT")
    print("==============================")

    print("Total candidate rows:", total_rows)
    print("Unique Source 1 IDs:", len(seen_ids))
    print("Duplicate Source 1 IDs:", duplicate_ids)

    if duplicate_ids == 0:
        print("PASS: No duplicate Source 1 IDs.")
    else:
        print("ERROR: Duplicate Source 1 IDs found.")

    # Check against ground truth
    print("\nChecking ground truth...")

    gt_ids = set()

    for chunk in pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        chunksize=100000
    ):
        gt_ids.update(chunk["source1_entity_id"])

    missing = gt_ids - seen_ids

    print("Ground truth Source 1 IDs:", len(gt_ids))
    print("Missing Source 1 IDs:", len(missing))

    if len(missing) == 0:
        print("PASS: Every ground-truth Source 1 ID is present.")
    else:
        print("ERROR: Some Source 1 IDs are missing.")


if __name__ == "__main__":
    validate_candidate_file()