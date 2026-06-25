import cv2
import numpy as np

from preprocessing.preprocessing_step import PreprocessingStep


class Resizer(PreprocessingStep):
    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
