import numpy as np
import oct_rust

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class RdchFeaturesExtractor(BaseFeaturesExtractor):
    """
    Ring Distributed Cumulative Histogram (RDCH) extractor.
    It divides the image into concentric rings radiating from a center point
    using a fixed step width, then creates a vector of intensities for each ring.
    """

    def __init__(
        self,
        ring_step: int,
        center: tuple[float, float] | None,
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the RDCH converter.

        Args:
            ring_step: The width of each concentric ring in pixels.
            center: The coordinates of the center point [0; 1]. If None, calculates center of mass.
            pixel_range: Tuple specifying valid intensities (min, max).
            bins: Number of bins to use for the histogram.
        """
        self.ring_step = ring_step

        # Split center tuple into separate args for the FFI
        self.center_x = center[0] if center is not None else None
        self.center_y = center[1] if center is not None else None

        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the RDCH features using parallelized Rust processing.
        """
        return oct_rust.extract_rdch(
            image,
            self.ring_step,
            self.center_x,
            self.center_y,
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
