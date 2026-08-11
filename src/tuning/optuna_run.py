import copy
import gc
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import optuna
import yaml
from sklearn.metrics import f1_score

from config_settings.oct_config import OctConfig
from dataset.oct_kfold_dataset import OctDatasetCrossVal
from pipeline.pipeline_builder import build_pipeline
from preprocessing.registry import PreprocessingRegistry

warnings.filterwarnings("ignore")


class FocusedOptunaTuner:
    """
    Optuna tuner that focuses on:
    - FIXED preprocessing (no tuning)
    - RCLDCH feature extractor (best performer)
    - Tunable classifiers (SVM with RBF, Random Forest)
    - PCA dimensionality reduction with wider search space
    """

    def __init__(
        self,
        config_path: Path,
        classifiers: list[str] | None = None,
        n_splits: int = 5,
        n_folds_to_train: int | None = None,
        validation_fold: int | None = None,
        random_state: int = 42,
    ):
        with open(config_path, "r") as file:
            self.base_config_dict = yaml.safe_load(file)

        base_config = OctConfig.model_validate(self.base_config_dict)

        preprocessing = PreprocessingRegistry.create_pipeline(
            base_config.preprocessing_pipeline
        )

        self.dataset = OctDatasetCrossVal(
            dataset_path=base_config.paths.original_dataset,
            allowed_classes=base_config.dataset.classes,
            samples_per_class=base_config.dataset.samples_per_class,
            preprocessor=preprocessing,
            n_splits=n_splits,
            n_folds_to_train=n_folds_to_train,
            validation_fold=validation_fold,
            shuffle=True,
            random_state=random_state,
        )

        self.classifiers = classifiers or ["svm", "random_forest"]
        self.random_state = random_state

    def _tune_rcldch_extractor(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune RCLDCH-specific parameters."""
        return {
            "sections": trial.suggest_categorical("rcldch_sections", [64, 128, 256]),
            "bins": trial.suggest_categorical("rcldch_bins", [128, 192, 256]),
            "pixel_range": (
                trial.suggest_int("rcldch_min_val", 0, 30),
                trial.suggest_int("rcldch_max_val", 200, 255),
            ),
        }

    def _tune_pca_reducer(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune PCA with wider component range."""
        return {
            "n_components": trial.suggest_int("pca_components", 80, 400, log=True),
            "whiten": trial.suggest_categorical("pca_whiten", [True, False]),
            "svd_solver": "randomized",
            "random_state": self.random_state,
            "iterated_power": trial.suggest_int("pca_iterated_power", 2, 8),
        }

    def _tune_svm_classifier(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune SVM with RBF kernel (non-linear)."""
        return {
            "kernel": "rbf",
            "C": trial.suggest_float("svm_c", 0.1, 100, log=True),
            "gamma": trial.suggest_float("svm_gamma", 1e-5, 0.1, log=True),
            "class_weight": trial.suggest_categorical("svm_class_weight", ["balanced"]),
            "shrinking": trial.suggest_categorical("svm_shrinking", [True, False]),
            "random_state": self.random_state,
            "max_iter": 2000,
        }

    def _tune_random_forest_classifier(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune Random Forest classifier."""
        bootstrap = trial.suggest_categorical("rf_bootstrap", [True, False])

        params = {
            "n_estimators": trial.suggest_int("rf_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 10, 40),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "rf_max_features", ["sqrt", "log2"]
            ),
            "bootstrap": bootstrap,
            "class_weight": trial.suggest_categorical(
                "rf_class_weight", ["balanced", "balanced_subsample"]
            ),
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        if bootstrap:
            params["max_samples"] = trial.suggest_float("rf_max_samples", 0.6, 1.0)

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Main Optuna objective function."""
        config_dict = copy.deepcopy(self.base_config_dict)

        # ============================================================
        # FIXED: Preprocessing (no tuning)
        # ============================================================
        # Preprocessing stays as-is from config

        # ============================================================
        # TUNE: RCLDCH Feature Extractor
        # ============================================================
        config_dict["feature_extractor"] = [
            {"rcldch": self._tune_rcldch_extractor(trial)}
        ]

        # ============================================================
        # TUNE: PCA Dimensionality Reduction
        # ============================================================
        config_dict["dimensionality_reduction"]["active"] = "pca"
        config_dict["dimensionality_reduction"]["params"] = self._tune_pca_reducer(
            trial
        )

        # ============================================================
        # TUNE: Classifier (SVM or Random Forest)
        # ============================================================
        active_classifier = trial.suggest_categorical(
            "classifier_type", self.classifiers
        )
        config_dict["classifier"]["active"] = active_classifier

        if active_classifier == "svm":
            config_dict["classifier"]["params"] = self._tune_svm_classifier(trial)
        elif active_classifier == "random_forest":
            config_dict["classifier"]["params"] = self._tune_random_forest_classifier(
                trial
            )

        config = OctConfig.model_validate(config_dict)

        # ============================================================
        # Evaluate on all k-folds
        # ============================================================
        fold_scores = []

        for fold_idx, train_indices, val_indices in self.dataset.get_folds():
            X_train, y_train = self.dataset.gather_fold_split(
                train_indices,
                description=f"Trial {trial.number} — Fold {fold_idx + 1} Train",
            )

            X_val, y_val = self.dataset.gather_fold_split(
                val_indices,
                description=f"Trial {trial.number} — Fold {fold_idx + 1} Validation",
            )

            pipeline = build_pipeline(
                config,
                image_preprocessing=True,
                feature_extracting=True,
                dimensionality_reduction=True,
                classification=True,
            )

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            fold_f1 = f1_score(y_val, predictions, average="macro")
            fold_scores.append(fold_f1)

            del pipeline, X_train, y_train, X_val, y_val, predictions
            gc.collect()

        mean_f1 = float(np.mean(fold_scores))

        # Report intermediate values for pruning
        trial.report(mean_f1, step=0)

        return mean_f1


def main():
    """Run focused Optuna optimization."""

    # Create or load study
    study = optuna.create_study(
        storage=os.environ.get("OPTUNA_DATABASE_URL", "sqlite:///optuna_tuning.db"),
        study_name="oct_rcldch_tuning",
        load_if_exists=True,
        direction="maximize",
    )

    # Create tuner
    tuner = FocusedOptunaTuner(
        config_path=Path("config/run.yaml"),
        classifiers=["svm", "random_forest"],
        n_splits=5,
        n_folds_to_train=None,  # Use all folds for final tuning
        random_state=42,
    )

    print("\n" + "=" * 70)
    print("FOCUSED OPTUNA TUNING: RCLDCH + PCA + SVM/RF")
    print("=" * 70)
    print(f"Study name: {study.study_name}")
    print(f"Direction: {study.direction.value}")
    print(f"Classifiers to tune: {tuner.classifiers}")
    print(f"Feature extractor: RCLDCH (fixed)")
    print(f"Preprocessing: Fixed (from config)")
    print("=" * 70 + "\n")

    # Run optimization
    study.optimize(
        tuner.objective,
        n_trials=100,
        n_jobs=1,  # Set to -1 for parallel if you have resources
        show_progress_bar=True,
    )

    # ================================================================
    # Results Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    best_trial = study.best_trial

    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Macro F1 Score: {best_trial.value:.4f}\n")

    print("Best Parameters:")
    for param_name, param_value in sorted(best_trial.params.items()):
        if isinstance(param_value, float):
            print(f"  {param_name:30s}: {param_value:.6f}")
        else:
            print(f"  {param_name:30s}: {param_value}")

    # ================================================================
    # Top 10 Trials
    # ================================================================
    print("\n" + "=" * 70)
    print("TOP 10 TRIALS")
    print("=" * 70)

    sorted_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    for i, trial in enumerate(sorted_trials[:10], 1):
        classifier = trial.params.get("classifier_type", "unknown")
        pca_comp = trial.params.get("pca_components", "?")
        f1 = trial.value or 0
        print(
            f"{i:2d}. Trial #{trial.number:4d} | {classifier:15s} | PCA={pca_comp:3} | F1={f1:.4f}"
        )

    # ================================================================
    # Optimization Statistics
    # ================================================================
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

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

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
