import numpy as np
from numpy.typing import NDArray

from preprocessing.preprocessing_step import PreprocessingStep


class RpeAligner(PreprocessingStep):
    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return image
