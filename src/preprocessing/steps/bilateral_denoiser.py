import cv2
import numpy as np

from preprocessing.preprocessing_step import PreprocessingStep


class BilateralDenoiser(PreprocessingStep):
    def __init__(self, d: int, sigma_color: int, sigma_space: int):
        """
        :param d: Diameter of pixel neighborhood (9 is standard for medical/moderate noise).
        :param sigma_color: The threshold for edge detection. Lower = stricter edge preservation.
        :param sigma_space: The physical distance to blur within.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        img_log = np.log1p(img)

        filtered = cv2.bilateralFilter(
            img_log,
            d=self.d,
            sigmaColor=self.sigma_color / 255.0,
            sigmaSpace=self.sigma_space,
        )

        result = np.expm1(filtered)
        result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        return result_uint8
