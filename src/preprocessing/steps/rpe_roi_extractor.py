import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from preprocessing.preprocessing_step import PreprocessingStep


class RpeRoiExtractor(PreprocessingStep):
    """
    Extracts a fixed-height ROI around the detected RPE.

    Unlike RpeAligner, this does not vertically shift individual columns.
    The natural curvature of the retina is therefore preserved.
    """

    def __init__(
        self,
        roi_above: int = 80,
        roi_below: int = 40,
        blur_sigma: int = 3,
        max_jump: int = 5,
    ) -> None:
        self.roi_above = roi_above
        self.roi_below = roi_below
        self.blur_sigma = blur_sigma
        self.max_jump = max_jump

    def preprocess(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        h, w = image.shape

        # ---------------------------------------------------------
        # 1. Smooth image for RPE detection
        # ---------------------------------------------------------

        if self.blur_sigma > 0:
            trace_image = gaussian_filter(
                image,
                sigma=self.blur_sigma,
            )
        else:
            trace_image = image

        # ---------------------------------------------------------
        # 2. Find initial RPE anchor
        # ---------------------------------------------------------

        max_intensities = np.max(
            trace_image,
            axis=0,
        )

        anchor_x = int(np.argmax(max_intensities))
        anchor_y = int(np.argmax(trace_image[:, anchor_x]))

        rpe_depths = np.zeros(
            w,
            dtype=np.int32,
        )

        rpe_depths[anchor_x] = anchor_y

        # ---------------------------------------------------------
        # 3. Trace RPE to the right
        # ---------------------------------------------------------

        current_y = anchor_y

        for x in range(anchor_x + 1, w):
            y_min = max(
                0,
                current_y - self.max_jump,
            )

            y_max = min(
                h,
                current_y + self.max_jump + 1,
            )

            window = trace_image[
                y_min:y_max,
                x,
            ]

            current_y = y_min + int(np.argmax(window))

            rpe_depths[x] = current_y

        # ---------------------------------------------------------
        # 4. Trace RPE to the left
        # ---------------------------------------------------------

        current_y = anchor_y

        for x in range(anchor_x - 1, -1, -1):
            y_min = max(
                0,
                current_y - self.max_jump,
            )

            y_max = min(
                h,
                current_y + self.max_jump + 1,
            )

            window = trace_image[
                y_min:y_max,
                x,
            ]

            current_y = y_min + int(np.argmax(window))

            rpe_depths[x] = current_y

        # ---------------------------------------------------------
        # 5. Extract ROI around RPE
        # ---------------------------------------------------------

        roi_height = self.roi_above + self.roi_below

        roi = np.zeros(
            (roi_height, w),
            dtype=np.uint8,
        )

        for x in range(w):
            rpe_y = rpe_depths[x]

            src_start = rpe_y - self.roi_above
            src_end = rpe_y + self.roi_below

            # Clip source coordinates to image boundaries.
            src_start_clipped = max(
                0,
                src_start,
            )

            src_end_clipped = min(
                h,
                src_end,
            )

            # Corresponding destination coordinates.
            dst_start = src_start_clipped - src_start
            dst_end = dst_start + (src_end_clipped - src_start_clipped)

            roi[
                dst_start:dst_end,
                x,
            ] = image[
                src_start_clipped:src_end_clipped,
                x,
            ]

        return roi
