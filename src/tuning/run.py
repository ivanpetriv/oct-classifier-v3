import argparse
import json
import os

import dotenv
import matplotlib
import numpy as np
import optuna

from tuning.early_stopping import EarlyStoppingCallback

matplotlib.use("Agg")
dotenv.load_dotenv()


from config_settings.oct_config import OctConfig
from tuning.objective import FocusedOptunaTuner
from tuning.search_spaces import generate_optuna_seed

# =============================================================
# BASE CONFIGURATION
# =============================================================


def create_study(
    db_path: str,
    study_name: str,
    seed: int = 42,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=1,
    )

    return optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=study_name,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


# =============================================================
# STARTING TRIAL
# =============================================================


def enqueue_start_trial(
    study: optuna.Study,
    start_values: dict,
) -> None:
    """Enqueue the known starting configuration.

    The starting configuration is only added to an empty study.
    Existing studies are never modified.
    """
    if not start_values:
        return

    if study.trials:
        return

    print("\n" + "=" * 60)
    print("INITIAL TUNING SEED")
    print("=" * 60)
    print(json.dumps(start_values, indent=4))
    print("=" * 60 + "\n")

    study.enqueue_trial(start_values)


# =============================================================
# RUN
# =============================================================


def run_study(
    *,
    config: OctConfig,
    db_path: str,
    study_name: str,
    n_trials: int,
    n_splits: int,
    n_folds: int | None,
    n_jobs: int,
    random_state: int,
    tune_preprocessing: bool,
    tune_extracting: bool,
    tune_pca: bool,
    tune_classification: bool,
    start_values: dict,
) -> optuna.Study:

    print()
    print("=" * 70)
    print("OPTUNA TUNING")
    print("=" * 70)

    print(f"Trials:                 {n_trials}")
    print(f"CV splits:              {n_splits}")
    print(f"Folds trained:          {n_folds if n_folds is not None else 'all'}")
    print(f"Tune preprocessing:     {tune_preprocessing}")
    print(f"Tune extractors:        {tune_extracting}")
    print(f"Tune PCA:               {tune_pca}")
    print(f"Tune classifier:        {tune_classification}")
    print(f"Parallel jobs:          {n_jobs}")
    print(f"Study:                  {study_name}")
    print("=" * 70)
    print()

    tuner = FocusedOptunaTuner(
        base_config=config,
        n_splits=n_splits,
        n_folds_to_train=n_folds,
        random_state=random_state,
        tune_preprocessing=tune_preprocessing,
        tune_extractors=tune_extracting,
        tune_pca=tune_pca,
        tune_classifier=tune_classification,
    )

    study = create_study(
        db_path=db_path,
        study_name=study_name,
        seed=random_state,
    )

    enqueue_start_trial(
        study=study,
        start_values=start_values,
    )

    study.optimize(
        tuner.objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[EarlyStoppingCallback(patience=100)],
    )

    os.system("paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null")

    return study


# =============================================================
# RESULTS
# =============================================================


def print_results(
    study: optuna.Study,
    title: str = "RESULTS",
) -> None:

    try:
        best_trial = study.best_trial
    except ValueError:
        print("No completed trials found.")
        return

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"Best Trial:       #{best_trial.number}")
    print(f"Best Macro F1:    {best_trial.value:.6f}")

    print()
    print("Best Parameters:")
    print("-" * 70)

    for key, value in sorted(best_trial.params.items()):
        print(f"{key:45s}: {value}")

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    values = [trial.value for trial in completed_trials if trial.value is not None]

    print()
    print("-" * 70)
    print("Statistics")
    print("-" * 70)

    print(f"Total trials:     {len(study.trials)}")
    print(f"Completed trials: {len(completed_trials)}")

    if values:
        print(f"Mean F1:          {np.mean(values):.6f}")
        print(f"Std F1:           {np.std(values):.6f}")
        print(f"Min F1:           {np.min(values):.6f}")
        print(f"Max F1:           {np.max(values):.6f}")

    print()
    print("-" * 70)
    print("Top 10 Trials")
    print("-" * 70)

    sorted_trials = sorted(
        completed_trials,
        key=lambda trial: trial.value if trial.value is not None else -np.inf,
        reverse=True,
    )

    print(
        f"{'Rank':<6}"
        f"{'Trial':<8}"
        f"{'Extractor':<15}"
        f"{'Reducer':<10}"
        f"{'Classifier':<15}"
        f"{'F1':<10}"
    )

    print("-" * 70)

    for rank, trial in enumerate(
        sorted_trials[:10],
        start=1,
    ):
        extractor = trial.params.get(
            "extractor_type",
            "fixed",
        )

        reducer = trial.params.get(
            "reducer_type",
            "fixed",
        )

        classifier = trial.params.get(
            "classifier_type",
            "fixed",
        )

        print(
            f"{rank:<6}"
            f"#{trial.number:<7}"
            f"{extractor:<15}"
            f"{reducer:<10}"
            f"{classifier:<15}"
            f"{trial.value:.6f}"
        )

    print("=" * 70)
    print()


# =============================================================
# EXPLORE STUDIES
# =============================================================


def get_studies(
    db_path: str,
) -> list[optuna.study.StudySummary]:
    storage_url = f"sqlite:///{db_path}"

    return optuna.study.get_all_study_summaries(
        storage=storage_url,
    )


def explore_studies(
    db_path: str,
) -> None:
    """List every Optuna study in the database."""

    storage_url = f"sqlite:///{db_path}"

    try:
        studies = get_studies(db_path)

    except Exception as exc:
        print(f"Failed to connect to database: {exc}")
        return

    if not studies:
        print(f"No studies found in {db_path}.")
        return

    print()
    print("=" * 100)
    print("OPTUNA STUDIES")
    print("=" * 100)

    print(
        f"{'Study':<35}{'Trials':<10}{'Direction':<12}{'Best F1':<12}{'Best Trial':<12}"
    )

    print("-" * 100)

    for summary in studies:
        if summary.best_trial is None:
            best_value = "N/A"
            best_number = "N/A"
        else:
            best_value = f"{summary.best_trial.value:.6f}"
            best_number = str(summary.best_trial.number)

        print(
            f"{summary.study_name:<35}"
            f"{summary.n_trials:<10}"
            f"{summary.direction.name:<12}"
            f"{best_value:<12}"
            f"{best_number:<12}"
        )

    print("=" * 100)
    print()


def explore_trials(
    db_path: str,
    study_name: str,
) -> None:
    """Print every trial in a specific study."""

    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(
            storage=storage_url,
            study_name=study_name,
        )

    except KeyError:
        print(f"\nStudy '{study_name}' does not exist in '{db_path}'.")
        print()

        explore_studies(db_path)
        return

    except Exception as exc:
        print(f"Failed to load study: {exc}")
        return

    print()
    print("=" * 100)
    print(f"TRIALS: {study.study_name}")
    print("=" * 100)

    print(f"Total trials: {len(study.trials)}")

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )

    pruned = sum(
        trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
    )

    failed = sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)

    print(f"Completed:    {completed}")
    print(f"Pruned:       {pruned}")
    print(f"Failed:       {failed}")

    print()
    print("-" * 100)

    sorted_trials = sorted(
        study.trials,
        key=lambda trial: trial.value if trial.value is not None else -np.inf,
        reverse=True,
    )

    for trial in sorted_trials:
        value = f"{trial.value:.6f}" if trial.value is not None else "N/A"

        print(f"Trial #{trial.number:<5} {trial.state.name:<10} F1={value}")

        if trial.params:
            for key, value in sorted(trial.params.items()):
                print(f"    {key}: {value}")

        print()

    print("=" * 100)


def analyze_results(
    db_path: str,
    study_name: str,
) -> None:
    storage_url = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(
            storage=storage_url,
            study_name=study_name,
        )
    except KeyError:
        available_studies = optuna.study.get_all_study_names(storage=storage_url)
        print(f"\n[ERROR] The study '{study_name}' does not exist in {db_path}.")
        print("-" * 50)
        print("Available studies in this database:")
        for name in available_studies:
            print(f"  - {name}")
        print("-" * 50)
        print(
            "Use the --study flag to select one. (e.g. --study oct_tuning_ldch_only)\n"
        )
        return
    except Exception as exc:
        print(f"Failed to connect to database: {exc}")
        return

    print_results(
        study,
        "STUDY ANALYSIS",
    )


# =============================================================
# ARGUMENTS
# =============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Optuna hyperparameter tuning for OCT feature extraction."),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "quick",
            "full",
            "analyze",
            "explore",
        ],
        default="quick",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials.",
    )

    parser.add_argument(
        "--explore-trials",
        action="store_true",
        help="Show all trials for the selected study.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help=("Number of folds used during optimization. Use 0 for all folds."),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna jobs.",
    )

    parser.add_argument(
        "--db",
        default="optuna_tuning.db",
        help="Optuna SQLite database.",
    )

    parser.add_argument(
        "--study",
        default="oct_full_tuning",
        help="Optuna study name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # ---------------------------------------------------------
    # TUNING TOGGLES
    # ---------------------------------------------------------

    preprocessing_group = parser.add_mutually_exclusive_group()

    preprocessing_group.add_argument(
        "--tune-preprocessing",
        dest="tune_preprocessing",
        action="store_true",
        help="Enable preprocessing hyperparameter tuning.",
    )

    preprocessing_group.add_argument(
        "--no-tune-preprocessing",
        dest="tune_preprocessing",
        action="store_false",
        help="Keep preprocessing fixed.",
    )

    parser.set_defaults(
        tune_preprocessing=True,
    )

    pca_group = parser.add_mutually_exclusive_group()

    pca_group.add_argument(
        "--tune-pca",
        dest="tune_pca",
        action="store_true",
        help="Enable PCA tuning.",
    )

    pca_group.add_argument(
        "--no-tune-pca",
        dest="tune_pca",
        action="store_false",
        help="Keep PCA fixed.",
    )

    ext_group = parser.add_mutually_exclusive_group()
    ext_group.add_argument(
        "--tune-extractors", dest="tune_extractors", action="store_true"
    )
    ext_group.add_argument(
        "--no-tune-extractors", dest="tune_extractors", action="store_false"
    )
    parser.set_defaults(tune_extractors=True)

    clf_group = parser.add_mutually_exclusive_group()
    clf_group.add_argument(
        "--tune-classifier", dest="tune_classifier", action="store_true"
    )
    clf_group.add_argument(
        "--no-tune-classifier", dest="tune_classifier", action="store_false"
    )
    parser.set_defaults(tune_classifier=True)

    parser.set_defaults(
        tune_pca=True,
    )

    return parser.parse_args()


# =============================================================
# MAIN
# =============================================================


def main() -> None:
    args = parse_args()

    if args.mode == "explore":
        if args.explore_trials:
            explore_trials(
                db_path=args.db,
                study_name=args.study,
            )
        else:
            explore_studies(
                db_path=args.db,
            )
        return

    if args.mode == "analyze":
        analyze_results(
            db_path=args.db,
            study_name=args.study,
        )
        return

    # ---------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------

    if args.mode == "quick":
        n_trials = args.trials if args.trials is not None else 10

        n_folds = args.folds if args.folds is not None else 1

    else:
        n_trials = args.trials if args.trials is not None else 150

        n_folds = args.folds if args.folds is not None else 0

    # 0 means all folds.
    n_folds_to_train = None if n_folds == 0 else n_folds

    # ---------------------------------------------------------
    # Base experiment configuration
    # ---------------------------------------------------------

    config = OctConfig.from_yaml("/home/ivan/University/thesis-oct-v3/config/run.yaml")

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    study = run_study(
        config=config,
        db_path=args.db,
        study_name=args.study,
        n_trials=n_trials,
        n_splits=5,
        n_folds=n_folds_to_train,
        n_jobs=args.jobs,
        random_state=args.seed,
        tune_preprocessing=args.tune_preprocessing,
        tune_extracting=args.tune_extractors,
        tune_pca=args.tune_pca,
        tune_classification=args.tune_classifier,
        start_values=generate_optuna_seed(),
    )

    print_results(
        study,
        "OPTUNA RESULTS",
    )

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
