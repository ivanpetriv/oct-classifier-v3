import gc
import os
import time
import warnings
from pathlib import Path

import yaml

from log.resource_tracker import track_trial_resources

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy

import optuna
from sklearn.metrics import f1_score

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

    def objective(self, trial: optuna.Trial) -> tuple[float, float, float, float]:
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

        with track_trial_resources() as train_res:
            pipeline.fit(self.X_train, self.y_train)

        train_preds = pipeline.predict(self.X_train)

        with track_trial_resources() as inf_res:
            val_preds = pipeline.predict(self.X_val)

        train_score = f1_score(self.y_train, train_preds, average="macro")
        val_score = f1_score(self.y_val, val_preds, average="macro")
        overfit_gap = max(0, train_score - val_score)

        inference_time = inf_res["time"]

        peak_memory = inf_res["peak_mem_mb"]

        del pipeline, config, config_dict, train_preds, val_preds
        gc.collect()

        return val_score, overfit_gap, inference_time, peak_memory


def main():
    study = optuna.create_study(
        storage="postgresql://postgres:34015221205343361642@localhost:5432/thesis-optuna",
        study_name="oct_tuning_all_v15",
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize", "minimize"],
        pruner=optuna.pruners.MedianPruner(),
    )

    # tuner = OptunaTuner(Path("config/run.yaml"))
    # study.optimize(tuner.objective, n_trials=5, n_jobs=1, timeout=3600)

    # 1. Get the list of all optimal trade-off trials

    pareto_front = study.best_trials

    print(
        f"Found {len(pareto_front)} optimal parameter configurations on the Pareto front:\n"
    )

    for i, trial in enumerate(pareto_front):
        print(f"--- Option {i + 1} (Trial #{trial.number}) ---")
        print(f"  Validation F1:    {trial.values[0]:.4f}")
        print(f"  Overfit Gap:      {trial.values[1]:.4f}")
        print(f"  Inference Time:   {trial.values[2]:.4f}s")
        print(f"  Peak Memory:      {trial.values[3]:.2f} MB")
        print(f"  Parameters:       {trial.params}\n")


if __name__ == "__main__":
    main()
