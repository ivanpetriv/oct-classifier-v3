import cv2
import numpy as np
from numpy.typing import NDArray

from preprocessing.preprocessing_step import PreprocessingStep


class CornerArtifactRemover(PreprocessingStep):
    def __init__(
        self,
        min_threshold: int,
        approximation_epsilon: float = 0.02,
        max_area_ratio: float = 0.15,
    ) -> None:
        self.min_threshold = min_threshold
        self.approximation_epsilon = approximation_epsilon
        self.max_area_ratio = max_area_ratio

    def preprocess(
        self,
        image: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        image = image.copy()
        img_h, img_w = image.shape

        _, bright_mask = cv2.threshold(
            image,
            self.min_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        contours, _ = cv2.findContours(
            bright_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        removal_mask = np.zeros(
            (img_h, img_w),
            dtype=np.uint8,
        )

        image_area = img_h * img_w

        for contour in contours:
            area = cv2.contourArea(contour)

            # Removed the `area == 0` check so thin vertical lines are not skipped
            if area > image_area * self.max_area_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Avoid division by zero or errors if the perimeter is exceptionally small
            if perimeter == 0:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                self.approximation_epsilon * perimeter,
                True,
            )

            # Added 2 to account for 1D line segments on the edges
            if len(polygon) not in (2, 3, 4, 5):
                continue

            x, y, w, h = cv2.boundingRect(polygon)

            touches_left = x <= 2
            touches_right = x + w >= img_w - 2
            touches_top = y <= 2
            touches_bottom = y + h >= img_h - 2

            touches_edge = (
                touches_left or touches_right or touches_top or touches_bottom
            )

            if not touches_edge:
                continue

            cv2.drawContours(
                removal_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )
            # Draw lines specifically for 2-point contours (thickness is needed since filled doesn't work on 1D lines)
            if len(polygon) == 2:
                cv2.drawContours(
                    removal_mask,
                    [contour],
                    -1,
                    255,
                    thickness=2,
                )

        # Dilate the mask slightly to catch the anti-aliased gradient edges of the white artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        removal_mask = cv2.dilate(removal_mask, kernel, iterations=1)

        image[removal_mask > 0] = 0

        return image
