#!/usr/bin/env python3
"""Train Matching Model & Tune Threshold on F0.5.

Workflow:
1. Loads train Source 1, Source 2, Source 3, and train ground truth.
2. Loads candidate pairs (if provided by Member 1) or generates a representative set of
   positive ground truth pairs + hard/soft negative candidate pairs.
3. Groups train/validation split by Source 1 entity ID to prevent data leakage.
4. Generates matching features for all pairs (name edit, token, Jaccard, address, numeric, country).
5. Trains a matching classifier (Random Forest default, modularly swappable with LightGBM).
6. Evaluates validation probabilities and tunes decision threshold to maximize F0.5.
7. Saves model, threshold configuration, and feature importance summary.

Usage:
    python scripts/train_matching.py
    python scripts/train_matching.py --model-type random_forest --num-samples 50000
    python scripts/train_matching.py --train-candidates output/train_candidate_pairs.tsv
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

# Add project root to sys.path so imports work seamlessly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    load_candidate_pairs,
    load_ground_truth,
    load_source_tsv,
    resolve_file_path,
)
from src.feature_engineering import (
    FEATURE_NAMES,
    build_entity_cache,
    compute_features_matrix,
)
from src.matching_model import MatchingClassifier
from src.text_processing import basic_clean
from src.threshold_tuning import evaluate_pair_predictions, find_optimal_threshold


class CombinedLookup:
    """Zero-copy combined lookup across Source 2 and Source 3 entity dictionaries."""
    def __init__(self, d1: Dict, d2: Dict):
        self.d1 = d1
        self.d2 = d2
    def get(self, k, default=None):
        v = self.d1.get(k)
        return v if v is not None else self.d2.get(k, default)
    def __getitem__(self, k):
        v = self.d1.get(k)
        if v is not None:
            return v
        return self.d2[k]
    def __contains__(self, k):
        return k in self.d1 or k in self.d2
    def keys(self):
        return list(self.d1.keys()) + list(self.d2.keys())
    def __len__(self):
        return len(self.d1) + len(self.d2)


def generate_training_candidate_pairs(
    ground_truth: Dict[str, Set[str]],
    source1_entities: Dict[str, Tuple],
    target_entities: Dict[str, Tuple],
    target_positives: int = 25000,
    neg_to_pos_ratio: float = 1.5,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[int], List[str]]:
    """Generate high-quality candidate pairs (positives + hard/soft negatives) for training."""
    rng = random.Random(random_seed)
    print("Generating representative training pairs from ground truth and source records...")

    # 1. Gather all positive pairs
    all_positive_pairs: List[Tuple[str, str]] = []
    for s1_id, true_targets in ground_truth.items():
        if s1_id not in source1_entities:
            continue
        for tid in true_targets:
            if tid in target_entities:
                all_positive_pairs.append((s1_id, tid))

    rng.shuffle(all_positive_pairs)
    if target_positives > 0 and len(all_positive_pairs) > target_positives:
        chosen_positives = all_positive_pairs[:target_positives]
    else:
        chosen_positives = all_positive_pairs

    n_pos = len(chosen_positives)
    n_neg = int(n_pos * neg_to_pos_ratio)
    print(f"Selected {n_pos:,} positive pairs. Generating {n_neg:,} negative candidate pairs...")

    # Index a representative sample of target entities for hard negative mining
    first_token_to_targets: Dict[str, List[str]] = {}
    all_target_ids = list(target_entities.keys())
    # Sample up to 150,000 targets for lightning-fast token indexing
    sample_target_ids = rng.sample(all_target_ids, min(150000, len(all_target_ids)))

    for tid in sample_target_ids:
        rec = target_entities.get(tid)
        if not rec:
            continue
        name = rec[0]
        cleaned = basic_clean(name)
        if cleaned:
            tok = cleaned.split()[0]
            if tok not in first_token_to_targets:
                first_token_to_targets[tok] = []
            if len(first_token_to_targets[tok]) < 25:
                first_token_to_targets[tok].append(tid)

    # 2. Hard and soft negative pair generation
    s1_ids_with_positives = list({p[0] for p in chosen_positives})
    chosen_negatives: List[Tuple[str, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set(chosen_positives)

    hard_neg_target = int(n_neg * 0.70)  # 70% hard negatives (same first token / similar name)
    soft_neg_target = n_neg - hard_neg_target

    # Mine hard negatives
    attempts = 0
    max_attempts = hard_neg_target * 5
    while len(chosen_negatives) < hard_neg_target and attempts < max_attempts:
        attempts += 1
        s1 = rng.choice(s1_ids_with_positives)
        rec = source1_entities.get(s1)
        if not rec or not rec[0]:
            continue
        cleaned_s1 = basic_clean(rec[0])
        if not cleaned_s1:
            continue
        tok = cleaned_s1.split()[0]
        candidates_pool = first_token_to_targets.get(tok)
        if not candidates_pool:
            continue
        cand_id = rng.choice(candidates_pool)
        pair = (s1, cand_id)
        if pair not in seen_pairs and cand_id not in ground_truth.get(s1, set()):
            seen_pairs.add(pair)
            chosen_negatives.append(pair)

    # Fill remainder with soft/random negatives
    while len(chosen_negatives) < n_neg:
        s1 = rng.choice(s1_ids_with_positives)
        cand_id = rng.choice(all_target_ids)
        pair = (s1, cand_id)
        if pair not in seen_pairs and cand_id not in ground_truth.get(s1, set()):
            seen_pairs.add(pair)
            chosen_negatives.append(pair)

    pairs = chosen_positives + chosen_negatives
    labels = [1] * len(chosen_positives) + [0] * len(chosen_negatives)
    s1_groups = [p[0] for p in pairs]

    # Combined shuffle
    combined = list(zip(pairs, labels, s1_groups))
    rng.shuffle(combined)
    pairs = [item[0] for item in combined]
    labels = [item[1] for item in combined]
    s1_groups = [item[2] for item in combined]

    print(f"Total candidate pairs generated: {len(pairs):,} ({sum(labels):,} positive, {len(labels) - sum(labels):,} negative)")
    return pairs, labels, s1_groups


def main():
    parser = argparse.ArgumentParser(description="Train Member 2 Matching Model & Tune Threshold on F0.5")
    parser.add_argument("--source1", default="train_source1.tsv", help="Path to train_source1.tsv")
    parser.add_argument("--source2", default="train_source2.tsv", help="Path to train_source2.tsv")
    parser.add_argument("--source3", default="train_source3.tsv", help="Path to train_source3.tsv")
    parser.add_argument("--ground-truth", default="train_ground_truth.tsv", help="Path to train_ground_truth.tsv")
    parser.add_argument("--train-candidates", default=None, help="Optional candidate pairs file for training")
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "lightgbm", "gradient_boosting"], help="Classifier architecture")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of training pairs (default: 50,000)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of decision trees (default: 100)")
    parser.add_argument("--max-depth", type=int, default=14, help="Maximum tree depth (default: 14)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation split ratio by entity (default: 0.20)")
    parser.add_argument("--output-model", default="models/matching_model.joblib", help="Output model path")
    parser.add_argument("--output-threshold", default="models/threshold_config.json", help="Output threshold JSON path")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Reconfigure stdout for UTF-8 display
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("MEMBER 2: MATCHING MODEL TRAINING & F0.5 THRESHOLD TUNING")
    print("=" * 70)

    # 1. Load Ground Truth
    print("\n[Step 1/6] Loading train ground truth...")
    gt_map = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(gt_map):,} Source 1 entities.")

    # 2. Load Source Datasets
    print("\n[Step 2/6] Loading training sources (Source 1, Source 2, Source 3)...")
    s1_dict = load_source_tsv(args.source1)
    print(f"  Source 1: {len(s1_dict):,} records loaded.")
    s2_dict = load_source_tsv(args.source2)
    print(f"  Source 2: {len(s2_dict):,} records loaded.")
    s3_dict = load_source_tsv(args.source3)
    print(f"  Source 3: {len(s3_dict):,} records loaded.")

    target_dict = CombinedLookup(s2_dict, s3_dict)
    print(f"  Total target records (S2 + S3): {len(target_dict):,}")

    # 3. Candidate Pairs & Labels
    print("\n[Step 3/6] Preparing candidate pairs and labels...")
    if args.train_candidates and os.path.isfile(args.train_candidates):
        print(f"Loading external candidate pairs from {args.train_candidates}...")
        flat_pairs, _, _ = load_candidate_pairs(args.train_candidates, max_pairs=args.num_samples if args.num_samples > 0 else None)
        pairs = flat_pairs
        labels = [1 if pair[1] in gt_map.get(pair[0], set()) else 0 for pair in pairs]
        s1_groups = [p[0] for p in pairs]
        print(f"Loaded {len(pairs):,} pairs ({sum(labels):,} positive, {len(labels) - sum(labels):,} negative).")
    else:
        target_pos = args.num_samples // 2 if args.num_samples > 0 else 25000
        pairs, labels, s1_groups = generate_training_candidate_pairs(
            ground_truth=gt_map,
            source1_entities=s1_dict,
            target_entities=target_dict,
            target_positives=target_pos,
            neg_to_pos_ratio=1.5,
            random_seed=args.random_seed,
        )

    # 4. Grouped Train/Validation Split by Source 1 Entity
    print("\n[Step 4/6] Splitting train & validation sets by Source 1 entity (grouped split)...")
    unique_s1 = sorted(list(set(s1_groups)))
    rng = random.Random(args.random_seed)
    rng.shuffle(unique_s1)

    n_val_s1 = int(len(unique_s1) * args.val_ratio)
    val_s1_set = set(unique_s1[:n_val_s1])
    train_s1_set = set(unique_s1[n_val_s1:])

    train_pairs, train_y = [], []
    val_pairs, val_y = [], []

    for pair, label in zip(pairs, labels):
        if pair[0] in val_s1_set:
            val_pairs.append(pair)
            val_y.append(label)
        else:
            train_pairs.append(pair)
            train_y.append(label)

    train_y_arr = np.array(train_y, dtype=int)
    val_y_arr = np.array(val_y, dtype=int)

    print(f"  Train set: {len(train_pairs):,} pairs ({sum(train_y):,} pos, {len(train_y) - sum(train_y):,} neg) across {len(train_s1_set):,} S1 entities.")
    print(f"  Val set:   {len(val_pairs):,} pairs ({sum(val_y):,} pos, {len(val_y) - sum(val_y):,} neg) across {len(val_s1_set):,} S1 entities.")

    # Build entity cache for feature extraction
    print("\nBuilding entity feature caches...")
    all_needed_s1 = {p[0] for p in pairs}
    all_needed_target = {p[1] for p in pairs}
    s1_cache = build_entity_cache({k: s1_dict[k] for k in all_needed_s1 if k in s1_dict})
    target_cache = build_entity_cache({k: target_dict[k] for k in all_needed_target if k in target_dict})

    print("Extracting feature matrices...")
    X_train = compute_features_matrix(train_pairs, s1_cache, target_cache)
    X_val = compute_features_matrix(val_pairs, s1_cache, target_cache)
    print(f"  X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # 5. Model Training
    print(f"\n[Step 5/6] Training {args.model_type} matching classifier...")
    model_params = {}
    if args.model_type == "random_forest":
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": "balanced",
            "random_state": args.random_seed,
            "n_jobs": -1,
        }
    elif args.model_type == "lightgbm":
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": min(args.max_depth, 10),
            "class_weight": "balanced",
            "random_state": args.random_seed,
            "n_jobs": -1,
        }

    clf = MatchingClassifier(
        model_type=args.model_type,
        feature_names=FEATURE_NAMES,
        model_params=model_params,
    )
    clf.fit(X_train, train_y_arr)
    print("Model fitting complete.")

    # 6. Threshold Tuning on F0.5
    print("\n[Step 6/6] Tuning decision threshold on validation set using F0.5...")
    val_probs = clf.predict_proba(X_val)

    best_thresh, best_f05, history = find_optimal_threshold(
        val_y_arr,
        val_probs,
        beta=0.5,
        min_threshold=0.10,
        max_threshold=0.95,
        step=0.01,
    )

    metrics_default = evaluate_pair_predictions(val_y_arr, val_probs, threshold=0.50, beta=0.5)
    metrics_optimal = evaluate_pair_predictions(val_y_arr, val_probs, threshold=best_thresh, beta=0.5)

    print("\n" + "-" * 70)
    print(f"EVALUATION SUMMARY ON VALIDATION SET:")
    print("-" * 70)
    print(f"  Default Threshold (0.50):")
    print(f"    Precision: {metrics_default['precision']:.4f}")
    print(f"    Recall:    {metrics_default['recall']:.4f}")
    print(f"    F0.5:      {metrics_default['f0.5']:.4f}")
    print(f"    F1:        {metrics_default['f1']:.4f}")
    print(f"    TP: {metrics_default['tp']:,} | FP: {metrics_default['fp']:,} | FN: {metrics_default['fn']:,} | TN: {metrics_default['tn']:,}")
    print()
    print(f"  Optimal Threshold ({best_thresh:.2f}) [Selected via F0.5 Optimization]:")
    print(f"    Precision: {metrics_optimal['precision']:.4f}  <-- Precision boosted for challenge scoring!")
    print(f"    Recall:    {metrics_optimal['recall']:.4f}")
    print(f"    F0.5:      {metrics_optimal['f0.5']:.4f}")
    print(f"    F1:        {metrics_optimal['f1']:.4f}")
    print(f"    TP: {metrics_optimal['tp']:,} | FP: {metrics_optimal['fp']:,} | FN: {metrics_optimal['fn']:,} | TN: {metrics_optimal['tn']:,}")
    print("-" * 70)

    # Feature importances
    print("\nTop 10 Feature Importances:")
    importances = clf.get_feature_importances()
    for feat_name, imp in list(importances.items())[:10]:
        print(f"  {feat_name:25s}: {imp:.4f}")

    # Save artifacts
    clf.save(args.output_model, threshold=best_thresh)
    print(f"\nSaved trained model: {args.output_model}")

    threshold_payload = {
        "selected_threshold": best_thresh,
        "metric_optimized": "F0.5",
        "validation_metrics": metrics_optimal,
        "default_metrics": metrics_default,
        "model_type": args.model_type,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_threshold)), exist_ok=True)
    with open(args.output_threshold, "w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)
    print(f"Saved threshold configuration: {args.output_threshold}")

    print("\n" + "=" * 70)
    print("TRAINING & THRESHOLD TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
