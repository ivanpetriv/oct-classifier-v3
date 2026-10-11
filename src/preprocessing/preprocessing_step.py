from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class PreprocessingStep(ABC):
    """
    Base class for preprocessing steps.
    """

    @abstractmethod
    def preprocess(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...
