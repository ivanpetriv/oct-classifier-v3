import copy
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from dataset.oct_dataset import OctDataset
from pipeline.cross_validation_runner import CrossValidationRunner

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import optuna
import yaml

from config_settings.oct_config import OctConfig
from preprocessing.registry import PreprocessingRegistry
from tuning.search_spaces import ClassifierTuners, FeatureExtractorTuners, ReducerTuners

warnings.filterwarnings("ignore")


class FocusedOptunaTuner:
    """
    Optuna tuner that focuses on:
    - FIXED preprocessing (no tuning)
    - RCLDCH feature extractor (best performer)
    - Tunable classifiers (SVM with RBF, Random Forest)
    - PCA dimensionality reduction with wider search space
    """

    def __init__(
        self,
        config_path: Path,
        classifiers: list[str] | None = None,
        n_splits: int = 5,
        n_folds_to_train: int | None = None,
        random_state: int = 42,
    ):
        with open(config_path, "r") as file:
            self.base_config_dict = yaml.safe_load(file)

        base_config = OctConfig.model_validate(self.base_config_dict)

        preprocessing = PreprocessingRegistry.create_pipeline(
            base_config.preprocessing_pipeline
        )

        self.dataset = OctDataset(
            dataset_path=base_config.paths.original_dataset,
            allowed_classes=base_config.dataset.classes,
            samples_per_class=base_config.dataset.samples_per_class,
            preprocessor=preprocessing,
        )

        self.n_splits = n_splits
        self.n_folds_to_train = n_folds_to_train
        self.classifiers = classifiers or ["svm", "random_forest"]
        self.random_state = random_state

    def objective(self, trial: optuna.Trial) -> float:
        """Main Optuna objective function."""
        config_dict = copy.deepcopy(self.base_config_dict)

        # TUNE: RCLDCH Feature Extractor
        config_dict["feature_extractor"] = [
            {"rcldch": FeatureExtractorTuners.tune_rcldch(trial)}
        ]

        # TUNE: PCA Dimensionality Reduction
        config_dict["dimensionality_reduction"]["active"] = "pca"
        config_dict["dimensionality_reduction"]["params"] = ReducerTuners.tune_pca(
            trial
        )

        # TUNE: Classifier
        active_classifier = trial.suggest_categorical(
            "classifier_type", self.classifiers
        )
        config_dict["classifier"]["active"] = active_classifier

        if active_classifier == "svm":
            config_dict["classifier"]["params"] = ClassifierTuners.tune_svm(trial)
        elif active_classifier == "random_forest":
            config_dict["classifier"]["params"] = ClassifierTuners.tune_random_forest(
                trial
            )

        # Validate the config
        config = OctConfig.model_validate(config_dict)

        # Utilize the CrossValidationRunner with verbose=False
        runner = CrossValidationRunner(
            config=config,
            dataset=self.dataset,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )

        mean_f1 = runner.run(n_folds_to_train=self.n_folds_to_train, verbose=False)

        return mean_f1
