import numpy as np
from numpy.typing import NDArray
from skimage.restoration import denoise_wavelet, estimate_sigma

from preprocessing.preprocessing_step import PreprocessingStep


class DwtDenoiser(PreprocessingStep):
    def __init__(
        self,
        wavelet: str,
        mode: str,
        wavelet_levels: int,
        method: str = "BayesShrink",
        rescale_sigma: bool = False,
        channel_axis: int | None = None,
    ):
        """
        Initializes the Discrete Wavelet Transform (DWT) denoiser.

        :param wavelet: The type of wavelet to use (e.g., 'sym4', 'db2', 'haar').
        :param mode: Thresholding mode ('soft' or 'hard'). Soft is generally better for preserving edges.
        :param wavelet_levels: Number of decomposition levels. Higher levels remove more low-frequency noise but can blur.
        :param method: Method to calculate the threshold ('BayesShrink' or 'VisuShrink').
        :param rescale_sigma: Whether to rescale the image variance before denoising (improves stability).
        :param channel_axis: Set to None for grayscale, or -1 for RGB images.
        """
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.method = method
        self.rescale_sigma = rescale_sigma
        self.channel_axis = channel_axis

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        img_float = image.astype(np.float32) / 255.0

        sigma_est = estimate_sigma(
            img_float, channel_axis=self.channel_axis, average_sigmas=True
        )

        clean_float = denoise_wavelet(
            img_float,
            wavelet=self.wavelet,
            mode=self.mode,
            sigma=sigma_est,
            wavelet_levels=self.wavelet_levels,
            method=self.method,
            rescale_sigma=self.rescale_sigma,
            channel_axis=self.channel_axis,
        )

        print(np.abs(img_float - clean_float))

        return np.clip(clean_float * 255.0, 0, 255).astype(np.uint8)
