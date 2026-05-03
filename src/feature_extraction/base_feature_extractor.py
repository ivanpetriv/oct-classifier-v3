from abc import ABC, abstractmethod
import numpy as np


class BaseFeaturesExtractor(ABC):
    """
    Base class for extracting features from an image.
    """
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an image to a histogram.
        :param image: Numpy 3D tensor representing image pixels (width*height*channels).
        :return: Numpy array representing the extracted features.
        """
        ...
