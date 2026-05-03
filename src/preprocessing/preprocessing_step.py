from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np


class PreprocessingStep(ABC):
    """
    Base class for preprocessing steps.
    """

    @abstractmethod
    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
