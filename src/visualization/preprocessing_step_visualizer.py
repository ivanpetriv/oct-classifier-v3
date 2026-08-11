import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from config_settings.oct_config import OctConfig
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.registry import PreprocessingRegistry


class PreprocessingStepsVisualizer:
    ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(self, pipeline: PreprocessingPipeline):
        self.pipeline = pipeline

    def visualize(
        self,
        image: np.ndarray,
        save_path: str | None = None,
    ):
        image = image.copy()

        image_stages: dict[str, np.ndarray] = {}
        image_stages["Original"] = image.copy()

        def capture_stage(step, index, image):
            image_stages[step.__class__.__name__] = image.copy()

        self.pipeline.add_hook("after_step", capture_stage)

        try:
            self.pipeline.execute(image)

            n_stages = len(image_stages)

            fig, axes = plt.subplots(
                1,
                n_stages,
                figsize=(4 * n_stages, 4),
            )

            # Make axes iterable when there is only one stage.
            axes = np.atleast_1d(axes)

            for ax, (title, display_array) in zip(
                axes,
                image_stages.items(),
            ):
                if display_array.dtype.kind in {"f", "c"}:
                    display_array = np.clip(
                        display_array,
                        0.0,
                        1.0,
                    )
                    display_array = (display_array * 255.0).astype(np.uint8)

                if display_array.ndim == 2:
                    ax.imshow(
                        display_array,
                        cmap="gray",
                    )
                else:
                    ax.imshow(display_array)

                ax.set_title(
                    title,
                    fontsize=12,
                    pad=10,
                )
                ax.axis("off")

            plt.tight_layout()

            if save_path is not None:
                plt.savefig(
                    save_path,
                    bbox_inches="tight",
                )

            plt.show()

        finally:
            self.pipeline.remove_hook("after_step")

    @classmethod
    def get_random_image(
        cls,
        dataset_path: Path,
        class_name: str | None = None,
    ) -> Path:
        """
        Returns a random image from the dataset.

        If class_name is provided, only images from that class
        are considered.
        """

        search_path = (
            dataset_path / class_name if class_name is not None else dataset_path
        )

        images = [
            path
            for path in search_path.rglob("*")
            if path.is_file() and path.suffix.lower() in cls.ALLOWED_EXTENSIONS
        ]

        if not images:
            raise ValueError(f"No images found in '{search_path}'.")

        return random.choice(images)

    def visualize_random(
        self,
        dataset_path: Path,
        class_name: str | None = None,
        save_path: str | None = None,
    ):
        """
        Selects a random image from the dataset and visualizes
        all preprocessing stages.
        """

        path = self.get_random_image(
            dataset_path,
            class_name,
        )

        print(f"Image: {path}")

        with Image.open(path) as image_obj:
            self.visualize(
                np.array(image_obj),
                save_path,
            )


if __name__ == "__main__":
    config = OctConfig.from_yaml("config/run.yaml")

    pipeline = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    visualizer = PreprocessingStepsVisualizer(pipeline)

    visualizer.visualize_random(
        dataset_path=config.paths.original_dataset,
        class_name="CNV",
    )
