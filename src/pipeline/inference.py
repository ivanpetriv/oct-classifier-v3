from pathlib import Path

import joblib  # type: ignore
import numpy as np
from PIL import Image

from config_settings.oct_config import OctConfig
from preprocessing.registry import PreprocessingRegistry


class Inferrer:
    """Takes a model and classifies an image with it."""

    def __init__(self, config: OctConfig, model_path: Path):
        self.config = config
        self.model_path = Path(model_path)

        self.pipeline = self._load_model()
        self.preprocessor = PreprocessingRegistry.create_pipeline(
            self.config.preprocessing_pipeline
        )

    def _load_model(self):
        """Internal method to handle safe model loading."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict(self, image_path: Path) -> str:
        """
        Loads an image, applies the exact training preprocessing,
        and returns the predicted class name.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert("L")
                image_array = np.array(img)

                if self.preprocessor is not None:
                    image_array = self.preprocessor.execute(image_array)

                image_array = np.ascontiguousarray(image_array, dtype=np.uint8)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load or process image from {image_path}: {e}"
            )

        # The pipeline expects a batch of images
        prediction_idx = self.pipeline.predict([image_array])[0]

        predicted_class = self.config.dataset.classes[int(prediction_idx)]
        return predicted_class

    def predict_array(self, image_array: np.ndarray) -> str:
        """Applies preprocessing and returns the predicted class name for an image array."""
        try:
            if self.preprocessor is not None:
                image_array = self.preprocessor.execute(image_array)

            image_array = np.ascontiguousarray(image_array, dtype=np.uint8)

        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {e}") from e

        prediction_idx = self.pipeline.predict([image_array])[0]

        predicted_class = self.config.dataset.classes[int(prediction_idx)]

        return predicted_class
