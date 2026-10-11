from typing import Literal
import numpy as np

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class LdchPyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Linear Distributed Cumulative Histogram (LDCH) extractor.
    It divides the image into horizontal or vertical sections,
    then creates a vector of intensities for each section.
    """

    def __init__(
        self,
        sections: int,
        layout: Literal["row", "col"],
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the LDCH converter.

        :param sections: Number of sections to divide the image into.
        :param layout: Layout of the sections, either "row" for horizontal or "col" for vertical.
        :param pixel_range: Tuple specifying the range of pixel values to consider (min, max).
        :param bins: Number of bins to use for the histogram.
        """
        if sections <= 0 or bins <= 0:
            raise ValueError
        if not (0 <= pixel_range[0] < pixel_range[1] <= 255):
            raise ValueError

        self.sections = sections
        self.layout = layout
        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        axis = 0 if self.layout == "row" else 1
        pixel_min, pixel_max = self.pixel_range
        pixel_range_size = int(pixel_max) - int(pixel_min) + 1

        chunks = np.array_split(image, self.sections, axis=axis)

        out_array = np.zeros((self.sections, self.bins), dtype=np.float64)

        for i, chunk in enumerate(chunks):
            flat = chunk.ravel()

            # Filter valid pixels within the specified range
            valid_mask = (flat >= pixel_min) & (flat <= pixel_max)
            valid_pixels = flat[valid_mask]
            total_valid = valid_pixels.size

            if total_valid > 0:
                normalized_pixels = valid_pixels.astype(np.int64) - pixel_min
                bin_indices = (normalized_pixels * self.bins) // pixel_range_size
                bin_indices = np.minimum(bin_indices, self.bins - 1)

                # Generate histogram counts
                counts = np.bincount(bin_indices, minlength=self.bins)

                # Calculate cumulative sum and normalize by total valid pixels
                out_array[i, :] = np.cumsum(counts) / total_valid

        return out_array
