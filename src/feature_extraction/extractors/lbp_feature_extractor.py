import numpy as np
from skimage.feature import local_binary_pattern

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class LbpFeaturesExtractor(BaseFeaturesExtractor):
    """
    Local Binary Patterns (LBP) feature extractor.
    Captures micro-textures of the tissue (edges, spots, flat areas).
    """

    def __init__(
        self, n_points: int, radius: int, method: str, norm: str, padding_threshold: int
    ):
        """
        Initializes the LBP extractor.
        :param n_points: Number of circularly symmetric neighbor set points.
        :param radius: Radius of circle.
        :param method: Method to determine the pattern ('uniform' reduces dimensionality).
        :param padding_threshold: Pixel intensity threshold to ignore black padding.
        """
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.norm = norm
        self.padding_threshold = padding_threshold

        if self.method == 'uniform':
            self.num_bins = self.n_points + 2
        else:
            self.num_bins = 2 ** self.n_points
    
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the LBP feature vector from the image.
        """
        lbp_image = local_binary_pattern(image, self.n_points, self.radius, self.method)
        
        # --- OPTIMIZATION: Ignore Background Padding ---
        tissue_mask = image > self.padding_threshold
        valid_lbp_values = lbp_image[tissue_mask]
        
        if len(valid_lbp_values) > 0:
            hist, _ = np.histogram(
                valid_lbp_values,
                bins=self.num_bins,
                range=(0, self.num_bins),
                density=True,
            )
        else:
            hist = np.zeros(self.num_bins, dtype=np.float64)
        
        return hist
