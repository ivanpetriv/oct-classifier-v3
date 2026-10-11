from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from config_settings.oct_config import OctConfig
from pipeline.pipeline_builder import build_pipeline


class Trainer:
    def __init__(self, config: OctConfig):
        self.config = config
        self.pipeline = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Builds and fits the machine learning pipeline."""
        self.pipeline = build_pipeline(
            self.config,
            True,
            True,
            True,
            True,
        )
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Cannot predict before training the pipeline.")
        return self.pipeline.predict(X_val)

    def save(self, file_path: str | Path) -> None:
        """Serializes and saves the trained pipeline to disk."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained pipeline.")

        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, str(path_obj))
