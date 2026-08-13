import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import optuna

from tuning.objective import FocusedOptunaTuner


def _create_and_run_study(
    mode_name: str,
    n_trials: int,
    n_folds: int,
    db_name: str,
    study_name: str,
    n_jobs: int = 1,
):
    print("\n" + "=" * 70)
    print(f"{mode_name.upper()} TUNING: {n_trials} trials × {n_folds} folds")
    print("=" * 70 + "\n")

    tuner = FocusedOptunaTuner(
        config_path=Path("config/run.yaml"),
        classifiers=["svm", "random_forest"],
        n_splits=5,
        n_folds_to_train=n_folds,
        random_state=42,
    )

    study = optuna.create_study(
        storage=f"sqlite:///{db_name}.db",
        study_name=study_name,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        tuner.objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    return study


def run_quick_test():
    """Quick test with 2 folds, 5 trials."""
    study = _create_and_run_study(
        mode_name="Quick Test",
        n_trials=5,
        n_folds=1,
        db_name="optuna_tuning_quick",
        study_name="oct_quick_test_v1",
        n_jobs=-1,
    )
    print_results(study, "QUICK TEST")


def run_full_tuning():
    """Full tuning with all 5 folds, 200 trials."""
    study = _create_and_run_study(
        mode_name="Full Tuning",
        n_trials=200,
        n_folds=1,
        db_name="optuna_tuning",
        study_name="oct_rcldch_tuning_v1",
        n_jobs=3,
    )
    print_results(study, "FULL TUNING")


def analyze_results(db_path: str = "sqlite:///optuna_tuning.db"):
    """Analyze existing results from previous runs."""
    print("\n" + "=" * 70)
    print("ANALYZING RESULTS")
    print("=" * 70 + "\n")

    try:
        study = optuna.load_study(
            storage=db_path,
            study_name="oct_rcldch_tuning",
        )
    except Exception as e:
        print(f"Failed to load study from {db_path}. Error: {e}")
        return

    print_results(study, "ANALYSIS")

    # Detailed breakdown by classifier
    print("\n" + "=" * 70)
    print("CLASSIFIER COMPARISON")
    print("=" * 70)

    trials_df = study.trials_dataframe()

    if "params_classifier_type" in trials_df.columns:
        for classifier in trials_df["params_classifier_type"].unique():
            scores = trials_df[
                (trials_df["params_classifier_type"] == classifier)
                & (trials_df["state"] == "COMPLETE")
            ]["value"]

            if len(scores) > 0:
                print(f"\n{classifier.upper()}:")
                print(f"  Trials: {len(scores)}")
                print(f"  Best: {scores.max():.4f}")
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Std:  {scores.std():.4f}")


def print_results(study: optuna.Study, title: str = "RESULTS"):
    """Print formatted results."""
    try:
        best_trial = study.best_trial
    except ValueError:
        print("No completed trials found in this study.")
        return

    print(f"\n{title}")
    print("=" * 70)
    print(f"Best Trial: #{best_trial.number}")
    print(f"Best Macro F1 Score: {best_trial.value:.4f}\n")

    print("Best Parameters:")
    print("-" * 70)

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
        description="Optuna tuning for RCLDCH OCT classifier"
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
