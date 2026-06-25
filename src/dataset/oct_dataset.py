import concurrent.futures
from pathlib import Path
from typing import Literal, Generator
import numpy as np
from PIL import Image

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from preprocessing.preprocessing_pipeline import PreprocessingPipeline


class OctDataset:
    allowed_image_extensions: set[str] = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        dataset_path: Path,
        allowed_classes: list[str],
        preprocessor: PreprocessingPipeline | None,
    ):
        self.path = dataset_path
        self.allowed_classes = allowed_classes
        self.preprocessor = preprocessor

    @property
    def class_mapping(self) -> dict[str, int]:
        return {class_name: idx for idx, class_name in enumerate(self.allowed_classes)}

    def _process_image(
        self, image_path: Path, label_idx: int, class_name: str
    ) -> tuple[np.ndarray, int, str]:
        with Image.open(image_path) as img:
            img = img.convert("L")
            arr = np.array(img)
            if self.preprocessor is not None:
                arr = self.preprocessor.execute(arr)
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        return arr, label_idx, class_name

    def _get_image_paths(
        self, split_path: Path, samples_per_class: int | None = None
    ) -> list[tuple[Path, int, str]]:
        image_paths: list[tuple[Path, int, str]] = []
        for class_name, label_idx in self.class_mapping.items():
            class_dir = split_path / class_name
            if not class_dir.exists():
                continue

            collected_for_class = 0
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in self.allowed_image_extensions:
                    image_paths.append((image_path, label_idx, class_name))
                    collected_for_class += 1

                    if (
                        samples_per_class is not None
                        and collected_for_class >= samples_per_class
                    ):
                        break

        if not image_paths:
            print(f"Warning: No images found for split '{split_path}'.")

        return image_paths

    def get_split_labels(
        self,
        dataset_split_name: Literal["train", "val", "test"],
        samples_per_class: int,
    ) -> np.ndarray:
        """Fast metadata pass: extracts labels without reading or processing images."""
        split_path = self.path / dataset_split_name
        image_paths = self._get_image_paths(split_path, samples_per_class)

        # Pull only the label integer (index 1) from the path metadata tuples
        labels = [label_idx for _, label_idx, _ in image_paths]
        return np.array(labels, dtype=np.uint8)

    def get_image_batch(
        self,
        dataset_split_name: Literal["train", "val", "test"],
        batch_size: int,
        samples_per_class: int | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        split_path = self.path / dataset_split_name
        image_paths = self._get_image_paths(split_path, samples_per_class)

        batch_images, batch_labels = [], []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.fields[current_class]}"),
            TimeRemainingColumn(),
        ) as progress:
            extraction_task = progress.add_task(
                f"[green]Extracting {dataset_split_name.upper()}",
                total=len(image_paths),
                current_class="",
            )

            current_displayed_class = ""

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_img = {
                    executor.submit(self._process_image, path, label, class_name): path
                    for path, label, class_name in image_paths
                }

                # As tasks complete (regardless of order), yield them
                for future in concurrent.futures.as_completed(future_to_img):
                    try:
                        image_array, label_idx, class_name = future.result()

                        del future_to_img[future]

                        if class_name != current_displayed_class:
                            progress.update(
                                extraction_task, current_class=f"({class_name})"
                            )
                            current_displayed_class = class_name

                        batch_images.append(image_array)
                        batch_labels.append(label_idx)

                        if len(batch_images) == batch_size:
                            yield (
                                np.array(batch_images, dtype=np.uint8),
                                np.array(batch_labels, dtype=np.uint8),
                            )
                            batch_images, batch_labels = [], []

                    except Exception as e:
                        failed_path = future_to_img[future]
                        progress.console.print(
                            f"[red]Failed to process {failed_path.name}: {e}"
                        )

                    progress.advance(extraction_task)

        # Yield remainder
        if batch_images:
            yield (
                np.array(batch_images, dtype=np.uint8),
                np.array(batch_labels, dtype=np.uint8),
            )

    def gather_full_split(
        self,
        dataset_split_name: Literal["train", "val", "test"],
        target_samples_per_class: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = 32
        generator = self.get_image_batch(
            dataset_split_name, batch_size, target_samples_per_class
        )

        X_batches, y_batches = [], []
        for X_batch, y_batch in generator:
            X_batches.append(X_batch)
            y_batches.append(y_batch)

        if not X_batches:
            raise ValueError(
                f"No data could be gathered for split '{dataset_split_name}'."
            )

        return (
            np.concatenate(X_batches, axis=0).astype(np.float32),
            np.concatenate(y_batches, axis=0),
        )
