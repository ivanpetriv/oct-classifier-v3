import copy
import os
from typing import Any

import numpy as np
import optuna

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from feature_extraction.registry import FeatureExtractorRegistry
from pipeline.cross_validation_runner import CrossValidationRunner
from preprocessing.registry import PreprocessingRegistry
from tuning.search_spaces import (
    ClassifierDefaults,
    ClassifierTuners,
    FeatureExtractorDefaults,
    FeatureExtractorTuners,
    PreprocessingDefaults,
    PreprocessingTuners,
    ReducerDefaults,
    ReducerTuners,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class FocusedOptunaTuner:
    def __init__(
        self,
        base_config: OctConfig,
        n_splits: int = 5,
        n_folds_to_train: int | None = None,
        random_state: int = 42,
        tune_preprocessing: bool = True,
        tune_extractors: bool = True,
        tune_pca: bool = True,
        tune_classifier: bool = True,
    ):
        self.base_config = base_config
        self.n_splits = n_splits
        self.n_folds_to_train = n_folds_to_train
        self.random_state = random_state

        self.tune_preprocessing = tune_preprocessing
        self.tune_extractors = tune_extractors
        self.tune_pca = tune_pca
        self.tune_classifier = tune_classifier

    # =========================================================
    # OBJECTIVE
    # =========================================================

    def objective(
        self,
        trial: optuna.Trial,
    ) -> float:
        try:
            config_dict = self.base_config.model_dump(
                mode="python",
            )

            config_dict["preprocessing_pipeline"] = self._configure_preprocessing(trial)

            config_dict["feature_extractor"] = self._configure_feature_extractors(trial)

            config_dict["dimensionality_reduction"] = self._configure_reducer(trial)

            config_dict["classifier"] = self._configure_classifier(trial)

            config = OctConfig.model_validate(
                config_dict,
            )

            preprocessing = PreprocessingRegistry.create_pipeline(
                config.preprocessing_pipeline,
            )

            dataset = OctDataset(
                dataset_path=config.paths.original_dataset,
                allowed_classes=config.dataset.classes,
                samples_per_class=config.dataset.samples_per_class,
                # preprocessor=preprocessing,
                preprocessor=None,
            )

            if trial.number == 0:
                self._check_feature_correlation(dataset)

            runner = CrossValidationRunner(
                config=config,
                dataset=dataset,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )

            return runner.run(
                n_folds_to_train=self.n_folds_to_train,
                verbose=False,
            )
        except ValueError as e:
            if "infs or NaNs" in str(e):
                trial.report(0.0, step=0)
                raise optuna.TrialPruned()

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _check_feature_correlation(self, dataset: OctDataset) -> None:
        try:
            row_ldch = FeatureExtractorRegistry.create(
                "ldch", sections=32, layout="row", pixel_range=[22, 202], bins=64
            )

            col_ldch = FeatureExtractorRegistry.create(
                "ldch", sections=32, layout="col", pixel_range=[37, 250], bins=64
            )

            # Load sample images
            sample_size = min(200, len(dataset.image_paths))
            sample_indices = np.arange(sample_size)
            X_images, _ = dataset.get_batch(
                sample_indices, description="Checking feature correlation"
            )

            # Extract features
            row_features = []
            col_features = []

            for image in X_images:
                row_features.append(row_ldch.extract(image).ravel())
                col_features.append(col_ldch.extract(image).ravel())

            row_features = np.array(row_features)
            col_features = np.array(col_features)

            # Filter out constant features (zero variance)
            row_std = np.std(row_features, axis=0)
            col_std = np.std(col_features, axis=0)

            row_mask = row_std > 1e-10
            col_mask = col_std > 1e-10

            row_features_filtered = row_features[:, row_mask]
            col_features_filtered = col_features[:, col_mask]

            print(f"\n{'=' * 60}")
            print(f"Row LDCH vs Col LDCH Correlation Check")
            print(
                f"Row features: {row_features.shape[1]} total, {row_features_filtered.shape[1]} with variance"
            )
            print(
                f"Col features: {col_features.shape[1]} total, {col_features_filtered.shape[1]} with variance"
            )

            # Compute correlation only on variable features
            if (
                row_features_filtered.shape[1] > 0
                and col_features_filtered.shape[1] > 0
            ):
                combined = np.hstack([row_features_filtered, col_features_filtered])
                correlation_matrix = np.corrcoef(combined.T)

                n_row = row_features_filtered.shape[1]
                cross_corr = correlation_matrix[:n_row, n_row:]

                mean_corr = np.abs(cross_corr).mean()
                max_corr = np.abs(cross_corr).max()

                print(f"Mean absolute correlation: {mean_corr:.3f}")
                print(f"Max correlation: {max_corr:.3f}")

                if mean_corr > 0.7:
                    print("⚠️  REDUNDANT - Skip Row+Col LDCH combination!")
                else:
                    print("✓ Good complementary info - Worth combining")
            else:
                print("⚠️  Too many constant features, cannot compute correlation")

            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"Could not compute correlation: {e}")

    def _configure_preprocessing(
        self,
        trial: optuna.Trial,
    ) -> list[dict[str, Any]]:
        configured_pipeline = []

        for index, step_name in enumerate(PreprocessingDefaults.ACTIVE):
            tuner = PreprocessingTuners.REGISTRY.get(step_name)

            if not self.tune_preprocessing or tuner is None:
                print("No specified params for preprocessing")
                static_params = copy.deepcopy(
                    PreprocessingDefaults.DEFAULTS.get(step_name, {})
                )
                configured_pipeline.append({step_name: static_params})
            else:
                tuned_params = tuner(
                    trial,
                    prefix=f"{step_name}_{index}",
                )
                configured_pipeline.append({step_name: tuned_params})

        return configured_pipeline

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def _configure_feature_extractors(
        self,
        trial: optuna.Trial,
    ) -> list[dict[str, Any]]:
        configured_extractors = []

        for index, step_dict in enumerate(FeatureExtractorDefaults.ACTIVE):
            for ext_name, base_params in step_dict.items():
                tuner = FeatureExtractorTuners.REGISTRY.get(ext_name)

                if not getattr(self, "tune_extractors", True) or tuner is None:
                    static_params = copy.deepcopy(base_params)
                    configured_extractors.append({ext_name: static_params})
                else:
                    tuned_params = tuner(
                        trial,
                        prefix=f"{ext_name}_{index}",
                    )
                    configured_extractors.append({ext_name: tuned_params})

        return configured_extractors

    # =========================================================
    # PCA
    # =========================================================

    def _configure_reducer(
        self,
        trial: optuna.Trial,
    ) -> dict[str, Any]:
        active = ReducerDefaults.ACTIVE

        if not active or active == "none":
            return {
                "active": "none",
                "params": {},
            }

        tuner = ReducerTuners.REGISTRY.get(active)

        if not self.tune_pca or tuner is None:
            static_params = copy.deepcopy(ReducerDefaults.DEFAULTS.get(active, {}))
            return {
                "active": active,
                "params": static_params,
            }

        return {
            "active": active,
            "params": tuner(
                trial,
                prefix=active,
            ),
        }

    # =========================================================
    # CLASSIFIER
    # =========================================================

    def _configure_classifier(self, trial: optuna.Trial) -> dict[str, Any]:
        active = ClassifierDefaults.ACTIVE
        if not active:
            return {"active": "none", "params": {}}

        tuner = ClassifierTuners.REGISTRY.get(active)

        # 🛠 FIX: Check self.tune_classifier
        if not self.tune_classifier or tuner is None:
            static_params = copy.deepcopy(ClassifierDefaults.DEFAULTS.get(active, {}))
            return {"active": active, "params": static_params}

        return {
            "active": active,
            "params": tuner(trial, prefix=active),
        }
