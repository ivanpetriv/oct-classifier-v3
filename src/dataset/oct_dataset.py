import concurrent.futures
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from preprocessing.preprocessing_pipeline import PreprocessingPipeline


class OctDataset:
    allowed_image_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(
        self,
        dataset_path: Path,
        allowed_classes: list[str],
        samples_per_class: int | None = None,
        preprocessor: PreprocessingPipeline | None = None,
    ):
        """Initializes the datsset.

        Args:
            dataset_path (Path): Path to the dataset. Must contain disease subfolders, and each subfolder contains images.
            allowed_classes (list[str]): Classes to load.
            samples_per_class (int | None, optional): How many items per disease. Defaults to None.
            preprocessor (PreprocessingPipeline | None, optional): Image preprocessor. Defaults to None.
        """
        self.path = dataset_path
        self.allowed_classes = allowed_classes
        self.samples_per_class = samples_per_class
        self.preprocessor = preprocessor

        # Maps disease names to numbers
        self.class_mapping = {c: i for i, c in enumerate(self.allowed_classes)}

        (
            self.image_paths,
            self.labels,
            self.class_names,
            self.patient_ids,
        ) = self._index_dataset()

    def _extract_patient_id(self, path: Path) -> str:
        """If using Ravelli dataset, extracts according to
        `<disease>-<patientid>-<number>`.
        Otherwise creates a random id.
        """
        match = re.match(r"^[^-]+-(\d+)-\d+$", path.stem)
        if match is None:
            return path.stem
        return match.group(1)

    def _index_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns image path, label index, class name, patient id.
        """
        image_paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        patient_ids: list[str] = []

        for class_name, label_idx in self.class_mapping.items():
            class_dir = self.path / class_name
            if not class_dir.exists():
                print(f"Warning: Disease directory '{class_dir}' does not exist.")
                continue

            valid_samples = 0
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in self.allowed_image_extensions:
                    continue
                if (
                    self.samples_per_class is not None
                    and valid_samples >= self.samples_per_class
                ):
                    break

                image_paths.append(image_path)
                labels.append(label_idx)
                class_names.append(class_name)
                patient_ids.append(self._extract_patient_id(image_path))

                valid_samples += 1

        return (
            np.array(image_paths),
            np.array(labels, dtype=np.uint8),
            np.array(class_names),
            np.array(patient_ids),
        )

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

    def _process_concurrently(
        self, paths: np.ndarray, labels: np.ndarray, classes: np.ndarray
    ):
        """Handles threading and yields results as they complete, completely blind to the UI."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_img = {
                executor.submit(self._process_image, p, l, c): p
                for p, l, c in zip(paths, labels, classes)
            }

            for future, val in future_to_img.items():
                try:
                    yield future.result(), None
                except Exception as e:
                    yield None, (val, e)

    def get_batch(
        self, indices: np.ndarray, description: str = "Loading Data"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads and processes the images for the given indices concurrently."""
        # This method only has rich progress bar.

        paths = self.image_paths[indices]
        labels = self.labels[indices]
        classes = self.class_names[indices]

        X_data: list[np.ndarray] = []
        y_data: list[int] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.fields[current_class]}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"[green]{description}", total=len(paths), current_class=""
            )
            current_displayed_class = ""

            for result, error in self._process_concurrently(paths, labels, classes):
                if error is not None:
                    path, e = error
                    progress.console.print(f"[red]Failed to process {path.name}: {e}")
                elif result is not None:
                    image_array, label_idx, class_name = result

                    if class_name != current_displayed_class:
                        progress.update(task, current_class=f"({class_name})")
                        current_displayed_class = class_name

                    X_data.append(image_array)
                    y_data.append(label_idx)

                progress.advance(task)

        if not X_data:
            raise ValueError("No data could be gathered for the given indices.")

        return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.uint8)
