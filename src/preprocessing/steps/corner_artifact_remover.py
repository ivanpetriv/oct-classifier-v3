import cv2
import numpy as np
from numpy.typing import NDArray

from preprocessing.preprocessing_step import PreprocessingStep


class CornerArtifactRemover(PreprocessingStep):
    def __init__(self, min_threshold: int) -> None:
        self.min_threshold = min_threshold

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        image = image.copy()
        img_h, img_w = image.shape[:2]

        _, bright_mask = cv2.threshold(
            image, self.min_threshold, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_to_remove = []

        for contour in contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)

            touches_top_left = x <= 2 and y <= 2
            touches_top_right = (x + w_box) >= img_w - 2 and y <= 2
            touches_bottom_left = x <= 2 and (y + h_box) >= img_h - 2
            touches_bottom_right = (x + w_box) >= img_w - 2 and (y + h_box) >= img_h - 2

            if (
                touches_top_left
                or touches_top_right
                or touches_bottom_left
                or touches_bottom_right
            ):
                contours_to_remove.append(contour)

        if contours_to_remove:
            removal_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(
                removal_mask, contours_to_remove, -1, 255, thickness=cv2.FILLED
            )

            # Expand the mask outward to swallow the anti-aliased halo
            kernel = np.ones((7, 7), np.uint8)
            removal_mask = cv2.dilate(removal_mask, kernel, iterations=1)

            # --- THE FIX: THE SAFE ZONE ---
            # Protect the middle 60% of the image (the retina) from being masked.
            # Even if the mask bleeds down the edge, it will be cut off here.
            safe_top = int(img_h * 0.20)
            safe_bottom = int(img_h * 0.80)
            removal_mask[safe_top:safe_bottom, :] = 0

            image[removal_mask == 255] = 0

        return image
