import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from preprocessing.preprocessing_step import PreprocessingStep


class RpeRoiExtractor(PreprocessingStep):
    def __init__(
        self,
        roi_above: int,
        roi_below: int,
        blur_sigma: int,
        max_jump: int,
        poly_degree: int,
    ) -> None:
        self.roi_above = roi_above
        self.roi_below = roi_below
        self.blur_sigma = blur_sigma
        self.max_jump = max_jump
        self.poly_degree = poly_degree

    def preprocess(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        h, w = image.shape
        # Smooth image
        if self.blur_sigma > 0:
            trace_image = gaussian_filter(
                image.astype(np.float32), sigma=self.blur_sigma
            )
        else:
            trace_image = image.astype(np.float32)

        # Bottom-weighting
        weights = (np.linspace(0.1, 1.0, h) ** 2).reshape(h, 1)
        weighted_image = trace_image * weights

        max_intensities = np.max(weighted_image, axis=0)
        anchor_x = int(np.argmax(max_intensities))
        anchor_y = int(np.argmax(weighted_image[:, anchor_x]))

        raw_depths = np.zeros(w, dtype=np.int32)
        raw_depths[anchor_x] = anchor_y

        # Trace right and left
        current_y = anchor_y
        for x in range(anchor_x + 1, w):
            y_min = max(0, current_y - self.max_jump)
            y_max = min(h, current_y + self.max_jump + 1)
            window = trace_image[y_min:y_max, x]

            if np.max(window) > 10.0:
                current_y = y_min + int(np.argmax(window))
            raw_depths[x] = current_y

        current_y = anchor_y
        for x in range(anchor_x - 1, -1, -1):
            y_min = max(0, current_y - self.max_jump)
            y_max = min(h, current_y + self.max_jump + 1)
            window = trace_image[y_min:y_max, x]

            if np.max(window) > 10.0:
                current_y = y_min + int(np.argmax(window))
            raw_depths[x] = current_y

        x_coords = np.arange(w)
        poly_coeffs = np.polyfit(x_coords, raw_depths, self.poly_degree)
        poly_func = np.poly1d(poly_coeffs)

        rpe_depths = np.clip(poly_func(x_coords), 0, h - 1).astype(np.int32)

        # Extract ROI
        roi_height = self.roi_above + self.roi_below
        roi = np.zeros((roi_height, w), dtype=np.uint8)

        for x in range(w):
            rpe_y = rpe_depths[x]
            src_start = rpe_y - self.roi_above
            src_end = rpe_y + self.roi_below

            src_start_clipped = max(0, min(h, src_start))
            src_end_clipped = max(0, min(h, src_end))

            dst_start = src_start_clipped - src_start
            dst_end = dst_start + (src_end_clipped - src_start_clipped)

            if src_start_clipped < src_end_clipped:
                roi[dst_start:dst_end, x] = image[src_start_clipped:src_end_clipped, x]

        return roi
