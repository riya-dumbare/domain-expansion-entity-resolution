# Dataset Setup

The Amazon ML Challenge dataset is not included in GitHub because the
dataset files are large.

Download the dataset from the Amazon ML Challenge 2026 challenge page and
extract it into the project root.

The expected structure is:

dataset/
├── test/
│   ├── test_source1.tsv
│   ├── test_source2.tsv
│   └── test_source3.tsv
├── train/
│   ├── train_ground_truth.tsv
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_source3.tsv
└── utils/
    └── validate_submission.py

Do not commit the dataset to GitHub.

The `dataset/` directory is already included in `.gitignore`.

After downloading and extracting the dataset, the blocking pipeline can be
run using the commands provided in the project README.