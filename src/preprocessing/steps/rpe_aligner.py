import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from preprocessing.preprocessing_step import PreprocessingStep


class RpeAligner(PreprocessingStep):
    def __init__(
        self,
        blur_sigma: int = 3,  # Lowered to a practical default
        poly_degree: int = 3,  # Lowered to prevent edge oscillation
        target_row: int | None = None,
        max_jump: int = 5,
    ):
        self.blur_sigma = blur_sigma
        self.poly_degree = poly_degree
        self.target_row = target_row
        self.max_jump = max_jump

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = image.shape
        raw_rpe_depths = np.zeros(w, dtype=int)

        # 1. Blur the image to reduce noise for the tracing logic
        if self.blur_sigma > 0:
            trace_image = gaussian_filter(image, sigma=self.blur_sigma)
        else:
            trace_image = image

        # 2. Find anchor on the smoothed image
        max_intensities = np.max(trace_image, axis=0)
        anchor_x = int(np.argmax(max_intensities))
        anchor_y = int(np.argmax(trace_image[:, anchor_x]))
        raw_rpe_depths[anchor_x] = anchor_y

        # 3. Trace Right
        current_y = anchor_y
        for x in range(anchor_x + 1, w):
            y_min = max(0, current_y - self.max_jump)  # Fixed wrap-around
            y_max = min(h, current_y + self.max_jump + 1)

            window = trace_image[y_min:y_max, x]
            local_max_idx = np.argmax(window)
            current_y = y_min + local_max_idx
            raw_rpe_depths[x] = current_y

        # 4. Trace Left
        current_y = anchor_y
        for x in range(anchor_x - 1, -1, -1):
            y_min = max(0, current_y - self.max_jump)  # Fixed wrap-around
            y_max = min(h, current_y + self.max_jump + 1)

            window = trace_image[y_min:y_max, x]
            local_max_idx = np.argmax(window)
            current_y = y_min + local_max_idx
            raw_rpe_depths[x] = current_y

        # 5. Fit polynomial
        x_coords = np.arange(w)
        poly_coeffs = np.polyfit(x_coords, raw_rpe_depths, self.poly_degree)
        poly_function = np.poly1d(poly_coeffs)

        # 6. Generate smooth curve
        smoothed_rpe_depths = np.round(poly_function(x_coords)).astype(int)

        # Bound the smoothed depths to image dimensions to prevent shift errors
        smoothed_rpe_depths = np.clip(smoothed_rpe_depths, 0, h - 1)

        align_y = h // 2 if self.target_row is None else self.target_row
        aligned_image = np.zeros_like(image)

        # 7. Shift raw image columns
        for x in range(w):
            shift = align_y - smoothed_rpe_depths[x]

            if shift > 0:
                src_start, src_end = 0, h - shift
                dst_start, dst_end = shift, h
            else:
                src_start, src_end = -shift, h
                dst_start, dst_end = 0, h + shift

            if src_start >= h or src_end <= 0 or dst_start >= h or dst_end <= 0:
                continue

            aligned_image[dst_start:dst_end, x] = image[src_start:src_end, x]

        return aligned_image
