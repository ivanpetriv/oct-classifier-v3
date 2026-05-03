import numpy as np
from skimage.feature import hog

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class HogFeaturesExtractor(BaseFeaturesExtractor):
    """
    Histogram of Oriented Gradients (HOG) feature extractor.
    Captures edge directions and physical structures.
    """
    
    def __init__(self,
                 orientations: int,
                 pixels_per_cell: tuple[int, int],
                 cells_per_block: tuple[int, int]):
        """
        Initializes the HOG extractor.
        :param orientations: Number of orientation bins.
        :param pixels_per_cell: Size of a cell (in pixels).
        :param cells_per_block: Number of cells in each block.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the HOG feature vector from the image.
        """        
        features = hog(image,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm='L2-Hys',
                       feature_vector=True)
        
        return features
