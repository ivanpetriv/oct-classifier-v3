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
        allowed_classes: list[str] | dict[str, str],
        samples_per_class: int | None = None,
        preprocessor: PreprocessingPipeline | None = None,
    ):
        """Initializes the dataset.

        Args:
            dataset_path (Path): Path to the dataset. Must contain disease subfolders.
            allowed_classes (list[str] | dict[str, str]): Classes to load.
                If a list, maps 1:1.
                If a dict, maps physical directory names to logical class names.
            samples_per_class (int | None, optional): How many items per logical class.
            preprocessor (PreprocessingPipeline | None, optional): Image preprocessor.
        """
        self.path = dataset_path
        self.samples_per_class = samples_per_class
        self.preprocessor = preprocessor

        if isinstance(allowed_classes, list):
            self.dir_to_logical = {c: c for c in allowed_classes}
        else:
            self.dir_to_logical = allowed_classes

        self.logical_classes = sorted(list(set(self.dir_to_logical.values())))

        self.class_mapping = {c: i for i, c in enumerate(self.logical_classes)}

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
        Pools directories into logical classes, then uses round-robin sampling
        to guarantee maximum patient variance across the combined data.
        """
        image_paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        patient_ids: list[str] = []

        # Structure: { logical_class: { patient_id: [path1, path2...] } }
        logical_to_patients: dict[str, dict[str, list[Path]]] = {
            c: {} for c in self.logical_classes
        }

        for dir_name, logical_class in self.dir_to_logical.items():
            class_dir = self.path / dir_name
            if not class_dir.exists():
                print(f"Warning: Disease directory '{class_dir}' does not exist.")
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in self.allowed_image_extensions:
                    continue

                patient_id = self._extract_patient_id(image_path)

                # Group under the logical class
                patient_dict = logical_to_patients[logical_class]
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                patient_dict[patient_id].append(image_path)

        # Execute Round-Robin Extraction per Logical Class
        for logical_class in self.logical_classes:
            label_idx = self.class_mapping[logical_class]
            patient_to_images = logical_to_patients[logical_class]

            valid_samples = 0
            active_patients = list(patient_to_images.keys())

            while active_patients:
                if (
                    self.samples_per_class is not None
                    and valid_samples >= self.samples_per_class
                ):
                    break

                for patient_id in list(active_patients):
                    if (
                        self.samples_per_class is not None
                        and valid_samples >= self.samples_per_class
                    ):
                        break

                    # Extract the next available image for this patient
                    img_path = patient_to_images[patient_id].pop(0)

                    image_paths.append(img_path)
                    labels.append(label_idx)
                    class_names.append(logical_class)
                    patient_ids.append(patient_id)

                    valid_samples += 1

                    # Remove the patient from the rotation if out of images
                    if not patient_to_images[patient_id]:
                        active_patients.remove(patient_id)

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
        """Handles threading and yields results as they complete."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_img = {
                executor.submit(self._process_image, p, l, c): p
                for p, l, c in zip(paths, labels, classes)
            }

            for future in concurrent.futures.as_completed(future_to_img):
                yield future.result()

    def get_batch(
        self, indices: np.ndarray, description: str = "Loading Data"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads and processes the images for the given indices concurrently."""

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

            for image_array, label_idx, class_name in self._process_concurrently(
                paths, labels, classes
            ):
                if class_name != current_displayed_class:
                    progress.update(task, current_class=f"({class_name})")
                    current_displayed_class = class_name

                X_data.append(image_array)
                y_data.append(label_idx)

                progress.advance(task)

        if not X_data:
            raise ValueError("No data could be gathered for the given indices.")

        return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.uint8)