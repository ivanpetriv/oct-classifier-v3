from typing import Literal
import numpy as np
import oct_rust

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class LdchFeaturesExtractor(BaseFeaturesExtractor):
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
        """
        Extracts the cumulative directional histogram from the given image.

        :param image: Image to extract features from.
        :return: A 2D numpy array containing normalized cumulative histograms for each section.
        """
        return oct_rust.extract_ldch(
            image,
            self.sections,
            self.layout == "row",
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
