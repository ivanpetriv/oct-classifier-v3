import copy
import multiprocessing
import os
from pathlib import Path

import optuna
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from classification.registry import ClassifierRegistry
from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from feature_extraction.registry import FeatureExtractorRegistry
from .setup import ClassifierTuners, FeatureExtractorTuners, ReducerTuners
from pipeline.train import TrainingPipeline


def objective(trial: optuna.Trial, config: OctConfig):
    config = copy.deepcopy(config)

    # Tune Extractor
    ext_active = config.feature_extractor.active
    if ext_active in FeatureExtractorTuners.REGISTRY:
        FeatureExtractorTuners.REGISTRY[ext_active](
            trial, config.feature_extractor.params
        )

    # Tune Reducer
    red_active = config.dimensionality_reduction.active
    if red_active in ReducerTuners.REGISTRY:
        ReducerTuners.REGISTRY[red_active](
            trial, config.dimensionality_reduction.params
        )

    # Tune Classifier
    clf_active = config.classifier.active
    if clf_active in ClassifierTuners.REGISTRY:
        ClassifierTuners.REGISTRY[clf_active](trial, config.classifier.params)

    feature_extractor = FeatureExtractorRegistry.create(
        config.feature_extractor.active, **vars(config.feature_extractor.params)
    )
    classifier = ClassifierRegistry.create(
        config.classifier.active, **vars(config.classifier.params)
    )

    dataset = OctDataset(config.paths.normalized_dataset)
    classes = config.dataset.classes

    match config.dimensionality_reduction.active:
        case "pca":
            pca = PCA(
                n_components=config.dimensionality_reduction.params.n_components,
                whiten=config.dimensionality_reduction.params.whiten,
                random_state=config.random_seed,
            )
            classifier = make_pipeline(pca, classifier)
        case _:
            pass

    # 3. Run the pipeline with the dynamically mutated config
    *_, results = TrainingPipeline(
        dataset, feature_extractor, classifier, classes, config.paths.model_output
    ).run()

    return results


def _worker_process(
    study_name: str, storage_name: str, base_config_path: Path, trials_per_worker: int
):
    """
    The isolated function that each parallel worker will execute.
    """
    # 1. Restrict CPU threads FOR THIS WORKER to prevent system freezing
    os.environ["RAYON_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"

    # 2. Connect to the existing shared database
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # 3. Run this worker's chunk of trials
    config = OctConfig.from_yaml(base_config_path)
    study.optimize(lambda trial: objective(trial, config), n_trials=trials_per_worker)


def launch_parallel_optimization():
    base_config = Path("config/run.yaml")
    storage_name = "sqlite:///oct_tuning.db"
    study_name = "thesis_optimization_v1"

    num_workers = 4  # How many parallel processes to spawn
    trials_per_worker = 1  # 4 workers * 12 trials = 48 total trials

    # 1. Initialize the database ONCE from the main process
    # This prevents race conditions where 4 workers try to create the DB at the exact same millisecond.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    print(f"[*] Study '{study_name}' created/loaded. Spawning {num_workers} workers...")

    # 2. Spawn the worker processes
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=_worker_process,
            args=(study_name, storage_name, base_config, trials_per_worker),
        )
        p.start()
        processes.append(p)
        print(f"    -> Worker {i + 1} started (PID: {p.pid})")

    # 3. Wait for all workers to finish
    for p in processes:
        p.join()

    print("\n[*] All workers finished.")

    # 4. Print the final aggregated best results
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print("\n--- Best Parameters Across All Workers ---")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best Metric Score: {study.best_value:.4f}")


if __name__ == "__main__":
    # In Windows/macOS, this block is strictly required for multiprocessing to work
    multiprocessing.set_start_method("spawn", force=True)
    launch_parallel_optimization()
