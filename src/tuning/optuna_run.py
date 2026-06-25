import gc
import os
from pathlib import Path
import time
import warnings

import yaml

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy

import optuna
from sklearn.metrics import classification_report, f1_score

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from pipeline.pipeline_builder import build_pipeline
from tuning.optuna_setup import ClassifierTuners, FeatureExtractorTuners, ReducerTuners

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
)


class OptunaTuner:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as file:
            self.base_config_dict = yaml.safe_load(file)

        base_config = OctConfig.model_validate(self.base_config_dict)

        self.dataset = OctDataset(
            base_config.paths.normalized_dataset, base_config.dataset.classes
        )
        self.X_train, self.y_train = self.dataset.gather_full_split("train", 1000)
        self.X_val, self.y_val = self.dataset.gather_full_split("val", 1000)

    def objective(self, trial: optuna.Trial) -> tuple[float, float, float]:
        config_dict = copy.deepcopy(self.base_config_dict)

        # --- TUNE EXTRACTOR ---
        ext_choice = trial.suggest_categorical(
            "active_extractor", list(FeatureExtractorTuners.REGISTRY.keys())
        )
        ext_params = FeatureExtractorTuners.REGISTRY[ext_choice](trial)
        config_dict["feature_extractor"]["active"] = ext_choice
        config_dict["feature_extractor"]["params"] = ext_params

        # --- TUNE REDUCER ---
        red_choice = trial.suggest_categorical(
            "active_reducer", list(ReducerTuners.REGISTRY.keys())
        )
        red_params = ReducerTuners.REGISTRY[red_choice](trial)
        config_dict["dimensionality_reduction"]["active"] = red_choice
        config_dict["dimensionality_reduction"]["params"] = red_params

        # --- TUNE CLASSIFIER ---
        clf_choice = trial.suggest_categorical(
            "active_classifier", list(ClassifierTuners.REGISTRY.keys())
        )
        clf_params = ClassifierTuners.REGISTRY[clf_choice](trial)
        config_dict["classifier"]["active"] = clf_choice
        config_dict["classifier"]["params"] = clf_params

        config = OctConfig.model_validate(config_dict)
        config.print_summary()

        pipeline = build_pipeline(config, True, True, True, True)

        pipeline.fit(self.X_train, self.y_train)

        train_preds = pipeline.predict(self.X_train)

        start_time = time.perf_counter()
        val_preds = pipeline.predict(self.X_val)
        finish_time = time.perf_counter()

        train_score = f1_score(self.y_train, train_preds, average="macro")
        val_score = f1_score(self.y_val, val_preds, average="macro")
        overfit_gap = max(0, train_score - val_score)

        del pipeline
        del config
        del config_dict
        del train_preds
        del val_preds
        gc.collect()

        return val_score, overfit_gap, finish_time - start_time


def main():
    study = optuna.create_study(
        storage="postgresql://postgres:34015221205343361642@localhost:5432/thesis-optuna",
        study_name="oct_tuning_all_v15",
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize"],
        pruner=optuna.pruners.MedianPruner(),
    )

    # tuner = OptunaTuner(Path("config/run.yaml"))
    # study.optimize(tuner.objective, n_trials=5, n_jobs=1, timeout=3600)

    # 1. Get the list of all optimal trade-off trials

    pareto_front = study.best_trials

    print(
        f"Found {len(pareto_front)} optimal parameter configurations on the Pareto front:\n"
    )

    # 2. Loop through them and print their specific metrics and parameters
    for i, trial in enumerate(pareto_front):
        print(f"--- Option {i + 1} (Trial #{trial.number}) ---")

        # trial.values contains the tuple of metrics you returned in the objective function
        print(f"  Validation F1: {trial.values[0]:.4f}")
        print(f"  Overfit Gap:   {trial.values[1]:.4f}")
        print(f"  Train Time:    {trial.values[2]:.2f}s")

        # trial.params contains the dictionary of hyperparameters used
        print(f"  Parameters:    {trial.params}\n")


if __name__ == "__main__":
    main()
