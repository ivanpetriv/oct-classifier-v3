"""
Quick start guide for running focused Optuna tuning.

Usage:
    python quick_start_optuna.py --mode quick     # 2-3 min test run
    python quick_start_optuna.py --mode full      # 1-2 hour full tuning
    python quick_start_optuna.py --mode analyze   # Analyze previous results
"""

import argparse
from pathlib import Path

import numpy as np
import optuna

from tuning.optuna_run import FocusedOptunaTuner


def run_quick_test():
    """Quick test with 2 folds, 20 trials (2-3 minutes)."""
    print("\n" + "=" * 70)
    print("QUICK TEST: 20 trials × 2 folds = ~3 minutes")
    print("=" * 70 + "\n")

    tuner = FocusedOptunaTuner(
        config_path=Path("config/run.yaml"),
        classifiers=["svm", "random_forest"],
        n_splits=5,
        n_folds_to_train=1,
        random_state=42,
    )

    study = optuna.create_study(
        storage="sqlite:///optuna_tuning_quick.db",
        study_name="oct_quick_test_v2",
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        tuner.objective,
        n_trials=5,
        n_jobs=-1,  # Parallel
        show_progress_bar=True,
    )

    print_results(study, "QUICK TEST")


def run_full_tuning():
    """Full tuning with all 5 folds, 150 trials (1-2 hours)."""
    print("\n" + "=" * 70)
    print("FULL TUNING: 150 trials × 5 folds = ~1-2 hours")
    print("=" * 70 + "\n")

    tuner = FocusedOptunaTuner(
        config_path=Path("config/run.yaml"),
        classifiers=["svm", "random_forest"],
        n_splits=5,
        n_folds_to_train=1,
        random_state=42,
    )

    study = optuna.create_study(
        storage="sqlite:///optuna_tuning.db",
        study_name="oct_rcldch_tuning",
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        tuner.objective,
        n_trials=1,
        n_jobs=3,
        show_progress_bar=True,
    )

    print_results(study, "FULL TUNING")


def analyze_results(db_path: str = "sqlite:///optuna_tuning.db"):
    """Analyze existing results from previous runs."""
    print("\n" + "=" * 70)
    print("ANALYZING RESULTS")
    print("=" * 70 + "\n")

    # Load study
    study = optuna.load_study(
        storage=db_path,
        study_name="oct_rcldch_tuning",
    )

    print_results(study, "ANALYSIS")

    # Detailed breakdown
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)

    trials_df = study.trials_dataframe()

    if "classifier_type" in trials_df.columns:
        for classifier in trials_df["classifier_type"].unique():
            scores = trials_df[
                (trials_df["classifier_type"] == classifier)
                & (trials_df["state"] == "COMPLETE")
            ]["value"]

            if len(scores) > 0:
                print(f"\n{classifier.upper()}:")
                print(f"  Trials: {len(scores)}")
                print(f"  Best: {scores.max():.4f}")
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Std:  {scores.std():.4f}")

    # PCA analysis
    print("\n" + "=" * 70)
    print("PCA COMPONENTS ANALYSIS")
    print("=" * 70)

    if "pca_components" in trials_df.columns:
        pca_df = trials_df[
            (trials_df["state"] == "COMPLETE") & (trials_df["value"].notna())
        ][["pca_components", "value"]].sort_values("pca_components")

        if len(pca_df) > 0:
            print("\nPCA Components vs F1 Score:")
            for _, row in pca_df.iterrows():
                pca_c = int(row["pca_components"])
                f1 = row["value"]
                print(f"  PCA={pca_c:3d}: F1={f1:.4f}")

    # RCLDCH analysis
    print("\n" + "=" * 70)
    print("RCLDCH CONFIGURATION ANALYSIS")
    print("=" * 70)

    if "rcldch_sections" in trials_df.columns:
        rcldch_df = trials_df[
            (trials_df["state"] == "COMPLETE") & (trials_df["value"].notna())
        ].copy()

        if len(rcldch_df) > 0:
            rcldch_df = (
                rcldch_df.groupby(["rcldch_sections", "rcldch_bins"])["value"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=False)
            )

            print("\nRCLDCH Configurations (Top 10):")
            print(rcldch_df.head(10).to_string())


def print_results(study, title: str = "RESULTS"):
    """Print formatted results."""
    best_trial = study.best_trial

    print(f"\n{title}")
    print("=" * 70)
    print(f"Best Trial: #{best_trial.number}")
    print(f"Best Macro F1 Score: {best_trial.value:.4f}\n")

    print("Best Parameters:")
    print("-" * 70)

    # Organize parameters by category
    params = best_trial.params
    categories = {
        "Classifier": [
            k for k in params if k.startswith(("svm_", "rf_", "classifier_"))
        ],
        "RCLDCH": [k for k in params if k.startswith("rcldch_")],
        "PCA": [k for k in params if k.startswith("pca_")],
    }

    for category, param_names in categories.items():
        if param_names:
            print(f"\n{category}:")
            for param in param_names:
                value = params[param]
                if isinstance(value, float):
                    print(f"  {param:35s}: {value:12.6f}")
                else:
                    print(f"  {param:35s}: {value}")

    # Statistics
    print("\n" + "-" * 70)
    print("Statistics:")
    print("-" * 70)

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    values = [t.value for t in completed_trials if t.value is not None]

    if values:
        print(f"Total trials completed: {len(completed_trials)}")
        print(f"Mean F1 score: {np.mean(values):.4f}")
        print(f"Std F1 score: {np.std(values):.4f}")
        print(f"Min F1 score: {np.min(values):.4f}")
        print(f"Max F1 score: {np.max(values):.4f}")
        print(f"Improvement over baseline (68%): {(np.max(values) - 0.68) * 100:.1f}%")

    # Top 10
    print("\n" + "-" * 70)
    print("Top 10 Trials:")
    print("-" * 70)

    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    print(f"{'Rank':<4} {'Trial':<8} {'Classifier':<15} {'PCA':<6} {'F1':<8}")
    print("-" * 70)

    for i, trial in enumerate(sorted_trials[:10], 1):
        classifier = trial.params.get("classifier_type", "?")
        pca_comp = int(trial.params.get("pca_components", 0))
        f1 = trial.value or 0
        print(f"{i:<4} #{trial.number:<7} {classifier:<15} {pca_comp:<6} {f1:.4f}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quick start Optuna tuning for RCLDCH OCT classifier"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "analyze"],
        default="quick",
        help="Mode: quick test (2-3 min), full tuning (1-2 hours), or analyze results",
    )

    args = parser.parse_args()

    if args.mode == "quick":
        run_quick_test()
    elif args.mode == "full":
        run_full_tuning()
    elif args.mode == "analyze":
        analyze_results()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
