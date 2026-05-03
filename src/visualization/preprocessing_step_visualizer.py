import numpy as np

from preprocessing.preprocessing_pipeline import PreprocessingPipeline


class PreprocessingStepsVisualizer:
    def __init__(self, pipeline: PreprocessingPipeline):
        self.pipeline = pipeline

    def visualize(self, image: np.ndarray, save_path: str | None = None): ...
