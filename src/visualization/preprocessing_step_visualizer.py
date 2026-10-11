import argparse
import random
from pathlib import Path
from typing import Any

import joblib  # type: ignore
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
        predicted_class: str | None = None,
        actual_class: str | None = None,
        confidence: float | None = None,
        font_scale: float = 1.0,
    ):
        """
        font_scale multiplies every font size (1.0 = large default).
        """

        def s(size: float) -> float:
            return size * font_scale

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
                figsize=(5.5 * n_stages, 6),
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
                    fontsize=s(22),
                    pad=14,
                )
                ax.axis("off")

            # Add prediction and confidence to the plot title if provided
            top = 1.0
            if predicted_class is not None:
                title_str = f"Prediction: {predicted_class}, actually: {actual_class}"
                if confidence is not None:
                    title_str += f" (Confidence: {confidence * 100:.1f}%)"
                fig.suptitle(title_str, fontsize=s(28), fontweight="bold")
                top = 0.90

            plt.tight_layout(rect=(0.0, 0.0, 1.0, top))

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
        model: Any | None = None,
        classes: list[str] | None = None,
        font_scale: float = 1.0,
    ):
        """
        Selects a random image from the dataset and visualizes
        all preprocessing stages.
        """
        if class_name is None:
            class_name = random.choice(classes or [])

        path = self.get_random_image(
            dataset_path,
            class_name,
        )

        print(f"Visualizing random image: {path}")

        with Image.open(path) as image_obj:
            img_array = np.array(image_obj.convert("L"))

            predicted_class = None
            confidence = None

            # Infer if a model was provided
            if model is not None and classes is not None:
                try:
                    preds = model.predict([img_array])
                    predicted_class = classes[int(preds[0])]
                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba([img_array])[0]
                        confidence = float(np.max(probas))
                except Exception as e:
                    print(f"Failed to infer with model: {e}")

            self.visualize(
                img_array,
                save_path,
                predicted_class=predicted_class,
                actual_class=class_name,
                confidence=confidence,
                font_scale=font_scale,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Preprocessing Pipeline Steps"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/run.yaml"),
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a specific image to visualize. If omitted, a random image will be used.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Target class for random image selection (default: CNV).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the trained model (.joblib). If provided, predicted class and confidence will be displayed.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the output matplotlib figure.",
    )

    args = parser.parse_args()

    # Load configuration and initialize pipeline
    config = OctConfig.from_yaml(args.config)
    pipeline = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)
    visualizer = PreprocessingStepsVisualizer(pipeline)

    # Load model if a path is provided
    model = None
    if args.model:
        if not args.model.exists():
            print(f"Error: Could not find model at {args.model}")
            exit(1)
        print(f"Loading model from: {args.model}")
        model = joblib.load(args.model)

    # Route logic based on whether an image was explicitly provided
    if args.image:
        if not args.image.exists():
            print(f"Error: Could not find image at {args.image}")
            exit(1)

        print(f"Visualizing provided image: {args.image}")
        with Image.open(args.image) as img:
            img_array = np.array(img.convert("L"))

            predicted_class = None
            confidence = None

            # Run inference if a model is available
            if model is not None:
                try:
                    preds = model.predict([img_array])
                    predicted_class = config.dataset.classes[int(preds[0])]
                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba([img_array])[0]
                        confidence = float(np.max(probas))
                except Exception as e:
                    print(f"Failed to infer with model: {e}")

            visualizer.visualize(
                img_array,
                save_path=args.save,
                predicted_class=predicted_class,
                actual_class=None,
                confidence=confidence,
            )
    else:
        visualizer.visualize_random(
            dataset_path=config.paths.original_dataset,
            class_name=args.class_name,
            save_path=args.save,
            model=model,
            classes=config.dataset.classes,
        )
