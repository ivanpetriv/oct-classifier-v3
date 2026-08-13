from pathlib import Path

import joblib  # type: ignore
import numpy as np
from sklearn.metrics import classification_report

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
        print(f"\n[Final Evaluation] Loading model from {self.model_path}...")
        pipeline = self._load_model()

        print("Loading test dataset...")
        preprocessing = PreprocessingRegistry.create_pipeline(
            self.config.preprocessing_pipeline
        )

        test_dataset = OctDataset(
            dataset_path=self.test_dataset_path,
            allowed_classes=self.config.dataset.classes,
            samples_per_class=None,
            preprocessor=preprocessing,
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

        class_mapping = test_dataset.class_mapping
        classes = list(class_mapping.keys())

        TrainResultsVisualizer.show_confusion_matrix(y_test, test_preds, classes)
