import pandas as pd
import os

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    path = os.path.join("data", file)

    print("\n==============================")
    print(file)
    print("==============================")

    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))