import cv2
import numpy as np

from preprocessing.preprocessing_step import PreprocessingStep


class ClaheFilter(PreprocessingStep):
    def __init__(self, clip_limit: float, tile_grid_size: tuple[int, int]):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ """
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        return clahe.apply(image)
