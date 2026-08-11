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

            if area == 0 or area > image_area * self.max_area_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)

            polygon = cv2.approxPolyDP(
                contour,
                self.approximation_epsilon * perimeter,
                True,
            )

            # Only triangles and quadrilaterals.
            if len(polygon) not in (3, 4):
                continue

            x, y, w, h = cv2.boundingRect(polygon)

            touches_top_left = x <= 2 and y <= 2
            touches_top_right = x + w >= img_w - 2 and y <= 2
            touches_bottom_left = x <= 2 and y + h >= img_h - 2
            touches_bottom_right = x + w >= img_w - 2 and y + h >= img_h - 2

            touches_corner = (
                touches_top_left
                or touches_top_right
                or touches_bottom_left
                or touches_bottom_right
            )

            if not touches_corner:
                continue

            # Remove exactly the detected polygon.
            cv2.drawContours(
                removal_mask,
                [polygon],
                -1,
                255,
                thickness=cv2.FILLED,
            )

        image[removal_mask > 0] = 0

        return image
