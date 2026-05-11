import numpy as np
import oct_rust

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor


class RcLdchFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        sections: int,
        pixel_range: tuple[int, int],
        bins: int,
    ):
        """
        Initializes the LDCH converter.

        :param sections: Number of sections to divide the image into.
        :param pixel_range: Tuple specifying the range of pixel values to consider (min, max).
        :param bins: Number of bins to use for the histogram.
        """
        if sections <= 0 or bins <= 0:
            raise ValueError(
                f"Sections and bins must be > 0. Got sections={sections}, bins={bins}"
            )
        if not (0 <= pixel_range[0] < pixel_range[1] <= 255):
            raise ValueError(
                f"Invalid pixel_range. Must be 0 <= min < max <= 255. Got {pixel_range}"
            )

        self.sections = sections
        self.pixel_range = pixel_range
        self.bins = bins

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the cumulative directional histogram from the given image.

        :param image: Image to extract features from.
        :return: A 2D numpy array containing normalized cumulative histograms for each section.
        """
        row_ldch = oct_rust.extract_ldch(
            image,
            self.sections,
            True,
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )
        col_ldch = oct_rust.extract_ldch(
            image,
            self.sections,
            False,
            self.pixel_range[0],
            self.pixel_range[1],
            self.bins,
        )

        return np.hstack(
            (
                row_ldch.ravel(),
                col_ldch.ravel(),
            )
        )
