import pandas as pd
from pathlib import Path


CANDIDATE_FILE = Path("output/candidate_pairs.tsv")
GROUND_TRUTH_FILE = Path("data/train_ground_truth.tsv")


def validate_candidate_file():
    if not CANDIDATE_FILE.exists():
        print("candidate_pairs.tsv not found.")
        print("Waiting for Member 1 to generate the file.")
        return

    df = pd.read_csv(CANDIDATE_FILE, sep="\t", dtype=str).fillna("")

    print("Candidate file loaded.")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required_columns = [
        "source1_entity_id",
        "candidate_entity_ids"
    ]

    if list(df.columns) != required_columns:
        print("ERROR: Wrong columns.")
        return

    if df["source1_entity_id"].duplicated().any():
        print("ERROR: Duplicate Source 1 IDs found.")
    else:
        print("PASS: No duplicate Source 1 IDs.")

    if GROUND_TRUTH_FILE.exists():
        gt = pd.read_csv(
            GROUND_TRUTH_FILE,
            sep="\t",
            dtype=str
        ).fillna("")

        missing = set(gt["source1_entity_id"]) - set(df["source1_entity_id"])

        print("Ground truth Source 1 IDs:", len(gt))
        print("Missing Source 1 IDs:", len(missing))

        if missing:
            print("ERROR: Some Source 1 IDs are missing.")
        else:
            print("PASS: Every ground-truth Source 1 ID is present.")


if __name__ == "__main__":
    validate_candidate_file()