import cv2
import numpy as np

from preprocessing.preprocessing_step import PreprocessingStep


class Cropper(PreprocessingStep):
    def __init__(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> None:
        """
        Initialize cropper with pixel margins to remove from each edge.

        Args:
            top: Pixels to remove from top
            bottom: Pixels to remove from bottom
            left: Pixels to remove from left
            right: Pixels to remove from right
        """
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image by removing fixed pixel margins from edges.

        Args:
            image: Input image array

        Returns:
            Cropped image
        """
        img_h, img_w = image.shape[:2]

        y_start = self.top
        y_end = img_h - self.bottom
        x_start = self.left
        x_end = img_w - self.right

        y_start = max(0, y_start)
        y_end = min(img_h, y_end)
        x_start = max(0, x_start)
        x_end = min(img_w, x_end)

        return image[y_start:y_end, x_start:x_end]
