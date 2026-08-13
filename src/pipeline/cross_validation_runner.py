import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from pipeline.train import Trainer
from visualization.train_results_visualizer import TrainResultsVisualizer


class CrossValidationRunner:
    """Runs the training loop with k-fold validation."""

    def __init__(
        self,
        config: OctConfig,
        dataset: OctDataset,
        n_splits: int,
        random_state: int,
    ):
        self.config = config
        self.dataset = dataset
        self.n_splits = n_splits
        self.random_state = random_state

    def run(self, n_folds_to_train: int | None = None, verbose: bool = True) -> float:
        """Executes the cross-validation loop and returns the mean macro F1 score."""
        if verbose:
            print(
                f"\n[K-Fold Cross-Validation]"
                f"\nSamples: {len(self.dataset.labels)}"
                f"\nTotal Folds: {self.n_splits}"
                f"\nSeed: {self.random_state}"
            )

        cv = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        all_splits = list(
            cv.split(
                self.dataset.image_paths,
                self.dataset.labels,
                groups=self.dataset.patient_ids,
            )
        )

        folds_to_run = list(range(self.n_splits))
        if n_folds_to_train is not None:
            folds_to_run = folds_to_run[:n_folds_to_train]

        all_val_targets = []
        all_val_preds = []

        self.config.paths.model_output.mkdir(parents=True, exist_ok=True)

        for fold_idx in folds_to_run:
            self._run_single_fold(
                fold_idx=fold_idx,
                splits=all_splits[fold_idx],
                all_val_targets=all_val_targets,
                all_val_preds=all_val_preds,
                verbose=verbose,
            )

        # Calculate the final score to return to Optuna
        overall_f1 = f1_score(all_val_targets, all_val_preds, average="macro")

        if verbose:
            self._evaluate_overall(all_val_targets, all_val_preds, len(folds_to_run))

        return float(overall_f1)

    def _run_single_fold(
        self,
        fold_idx: int,
        splits: tuple[np.ndarray, np.ndarray],
        all_val_targets: list,
        all_val_preds: list,
        verbose: bool,
    ) -> None:
        """Handles the lifecycle of a single fold, including creating a fresh model."""
        train_indices, val_indices = splits

        self._check_leakage(train_indices, val_indices)

        if verbose:
            print(
                f"\n==================== FOLD {fold_idx + 1}/{self.n_splits} ===================="
            )

        X_train, y_train = self.dataset.get_batch(
            train_indices, description=f"Fold {fold_idx + 1} Train"
        )
        X_val, y_val = self.dataset.get_batch(
            val_indices, description=f"Fold {fold_idx + 1} Val"
        )

        trainer = Trainer(self.config)
        trainer.train(X_train, y_train)

        val_preds = trainer.predict(X_val)

        if verbose:
            print(classification_report(y_val, val_preds, zero_division=0))

        all_val_targets.extend(y_val)
        all_val_preds.extend(val_preds)

    def _check_leakage(
        self, train_indices: np.ndarray, val_indices: np.ndarray
    ) -> None:
        """Ensures patients in the validation set do not appear in the training set."""
        train_patients = set(self.dataset.patient_ids[train_indices])
        val_patients = set(self.dataset.patient_ids[val_indices])
        overlap = train_patients & val_patients

        if overlap:
            raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:10]}")

    def _evaluate_overall(
        self, all_val_targets: list, all_val_preds: list, num_folds_run: int
    ) -> None:
        """Calculates and prints the aggregated metrics across all executed folds."""
        if num_folds_run <= 1:
            return

        print("\n==================== OVERALL K-FOLD RESULTS ====================")
        targets_arr = np.asarray(all_val_targets)
        preds_arr = np.asarray(all_val_preds)

        print(classification_report(targets_arr, preds_arr, zero_division=0))

        TrainResultsVisualizer.show_confusion_matrix(
            targets_arr,
            preds_arr,
            list(self.dataset.class_mapping.keys()),
        )
