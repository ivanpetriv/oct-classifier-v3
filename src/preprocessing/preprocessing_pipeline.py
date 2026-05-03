from typing import Any, Callable

import numpy as np

from preprocessing.preprocessing_step import PreprocessingStep


class PreprocessingPipeline:
    def __init__(
        self,
        steps: list[PreprocessingStep],
        hooks: dict[str, list[Callable[..., None]]] | None = None,
    ):
        self.steps = steps
        self.hooks = hooks or {}

    def _emit(self, name: str, **data: Any) -> None:
        for fn in self.hooks.get(name, []):
            fn(**data)

    def execute(self, image_array: np.ndarray) -> np.ndarray:
        """
        Executes image preprocessing on a given image array.

        :param image_array: NumPy array.
        :return: A preprocessed NumPy array.
        """

        for i, step in enumerate(self.steps):
            image_array = step.preprocess(image_array)

            self._emit("after_step", step=step, index=i, image=image_array)

        if image_array.dtype.kind in {"f", "c"}:
            image_array = (image_array * 255.0).astype(np.uint8)

        return image_array
