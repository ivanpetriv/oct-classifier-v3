from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from preprocessing.registry import PreprocessingRegistry
from visualization.train_results_visualizer import TrainResultsVisualizer


class Evaluator:
    def __init__(
        self, config: OctConfig, model_path: str | Path, test_dataset_path: str | Path
    ):
        self.config = config
        self.model_path = Path(model_path)
        self.test_dataset_path = Path(test_dataset_path)

    def _load_model(self):
        """Internal method to safely load the trained pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def evaluate(self) -> None:
        """
        Loads the test dataset, runs predictions using the saved model,
        and generates performance reports and visualizations.
        """
        pipeline = self._load_model()

        preprocessing = PreprocessingRegistry.create_pipeline(
            self.config.preprocessing_pipeline
        )

        test_dataset = OctDataset(
            dataset_path=self.test_dataset_path,
            allowed_classes=self.config.dataset.classes,
            samples_per_class=None,
            # preprocessor=preprocessing,
            preprocessor=None,
        )

        total_samples = len(test_dataset.labels)
        if total_samples == 0:
            raise ValueError("Test dataset is empty or paths are incorrect.")

        all_indices = np.arange(total_samples)
        X_test, y_test = test_dataset.get_batch(
            all_indices, description="Loading Test Data"
        )

        print("\nRunning predictions...")
        test_preds = pipeline.predict(X_test)

        print("\n==================== TEST SET RESULTS ====================")
        print(classification_report(y_test, test_preds, zero_division=0))

        print("\n================= 95% CONFIDENCE INTERVALS =================")
        print("Calculating bootrapped intervals (this may take a few seconds)...")
        cis = self._calculate_bootstrap_cis(y_test, test_preds)
        for metric, (lower, upper) in cis.items():
            print(f"{metric:<18}: [{lower:.3f}, {upper:.3f}]")
        print("==========================================================\n")

        class_mapping = test_dataset.class_mapping
        classes = list(class_mapping.keys())

        paths_column = test_dataset.image_paths

        # Save image-by-image results
        results_df = pd.DataFrame(
            {
                "image_path": paths_column,  # Ensures exact alignment between models
                "y_true": y_test,
                "y_pred": test_preds,
            }
        )

        model_name = self.model_path.stem
        output_file = self.config.paths.model_output / f"1_{model_name}_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Detailed predictions saved for McNemar's test: {output_file}")

        TrainResultsVisualizer.show_confusion_matrix(y_test, test_preds, classes)

    def _calculate_bootstrap_cis(
        self, y_true, y_pred, n_bootstraps=1000, ci_level=0.95
    ):
        """Internal method to calculate empirical confidence intervals via bootstrapping."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        n_samples = len(y_true)
        classes = np.unique(y_true)

        f1_scores = []
        accuracies = []
        specificities = []

        np.random.seed(42)  # Ensures reproducible CI results across multiple runs

        for _ in range(n_bootstraps):
            indices = np.random.randint(0, n_samples, n_samples)
            sample_true = y_true[indices]
            sample_pred = y_pred[indices]

            # F1 and Accuracy
            f1 = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
            acc = accuracy_score(sample_true, sample_pred)

            # Multi-class Macro Specificity
            cm = confusion_matrix(sample_true, sample_pred, labels=classes)
            spec_per_class = []

            for i in range(len(classes)):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                tn = np.sum(cm) - (tp + fp + fn)

                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                spec_per_class.append(specificity)

            macro_specificity = np.mean(spec_per_class)

            f1_scores.append(f1)
            accuracies.append(acc)
            specificities.append(macro_specificity)

        lower_percentile = ((1.0 - ci_level) / 2.0) * 100
        upper_percentile = (ci_level + ((1.0 - ci_level) / 2.0)) * 100

        return {
            "Accuracy": (
                np.percentile(accuracies, lower_percentile),
                np.percentile(accuracies, upper_percentile),
            ),
            "Macro F1-Score": (
                np.percentile(f1_scores, lower_percentile),
                np.percentile(f1_scores, upper_percentile),
            ),
            "Macro Specificity": (
                np.percentile(specificities, lower_percentile),
                np.percentile(specificities, upper_percentile),
            ),
        }
