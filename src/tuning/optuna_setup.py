from typing import Any

import optuna


class FeatureExtractorTuners:
    @staticmethod
    def tune_hog(trial: optuna.Trial) -> dict[str, Any]:
        cell_size = trial.suggest_categorical("hog_cell_size", [8, 16, 32])
        block_size = trial.suggest_categorical("hog_cells_per_block", [1, 2, 3])

        return {
            "orientations": trial.suggest_int("hog_orientations", 6, 12),
            "pixels_per_cell": (cell_size, cell_size),
            "cells_per_block": (block_size, block_size),
            "transform_sqrt": trial.suggest_categorical(
                "hog_transform_sqrt", [True, False]
            ),
            "block_norm": trial.suggest_categorical(
                "hog_block_norm", ["L1", "L1-sqrt", "L2", "L2-Hys"]
            ),
        }

    @staticmethod
    def tune_lbp(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_points": trial.suggest_int("lbp_n_points", 8, 24, step=8),
            "radius": trial.suggest_int("lbp_radius", 1, 4),
            "method": trial.suggest_categorical(
                "lbp_method", ["default", "uniform", "nri_uniform"]
            ),
            "norm": trial.suggest_categorical("lbp_norm", ["l1", "l2"]),
            "padding_threshold": trial.suggest_int("lbp_padding_threshold", 0, 50),
        }

    @staticmethod
    def tune_cdch(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "angle_step": trial.suggest_int("cdch_angle_step", 4, 45),
            "bins": trial.suggest_int("cdch_bins", 32, 256, step=32),
            "pixel_range": (
                trial.suggest_int("cdch_min_val", 0, 50),
                trial.suggest_int("cdch_max_val", 200, 255),
            ),
            "center": (
                trial.suggest_float("cdch_center_x", 0.1, 0.9),
                trial.suggest_float("cdch_center_y", 0.1, 0.9),
            ),
        }

    @staticmethod
    def tune_ldch(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "sections": trial.suggest_categorical("dch_sections", [32, 64, 128]),
            "bins": trial.suggest_int("dch_bins", 64, 256, step=64),
            "pixel_range": (
                trial.suggest_int("ldch_min_val", 0, 50),
                trial.suggest_int("ldch_max_val", 200, 255),
            ),
            "layout": trial.suggest_categorical("ldch_layout", ["row", "col"]),
        }

    @staticmethod
    def tune_rdch(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "ring_step": trial.suggest_int("rdch_ring_step", 4, 128),
            "bins": trial.suggest_int("rdch_bins", 64, 256, step=64),
            "pixel_range": (
                trial.suggest_int("rdch_min_val", 0, 50),
                trial.suggest_int("rdch_max_val", 200, 255),
            ),
            "center": (
                trial.suggest_float("rdch_center_x", 0.1, 0.9),
                trial.suggest_float("rdch_center_y", 0.1, 0.9),
            ),
        }

    @staticmethod
    def tune_rcldch(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "bins": trial.suggest_int("rcldch_bins", 64, 256, step=64),
            "sections": trial.suggest_categorical("rcldch_sections", [32, 64, 128]),
            "pixel_range": (
                trial.suggest_int("rcldch_min_val", 0, 50),
                trial.suggest_int("rcldch_max_val", 200, 255),
            ),
        }

    REGISTRY = {
        # "hog": tune_hog,
        # "lbp": tune_lbp,
        "cdch": tune_cdch,
        "ldch": tune_ldch,
        "rdch": tune_rdch,
        "rcldch": tune_rcldch,
    }


class ReducerTuners:
    SOLVER_CHOICES = ["randomized"]
    NORMALIZER_CHOICES = ["auto", "QR", "LU", "none"]

    @staticmethod
    def tune_pca(trial: optuna.Trial) -> dict[str, Any]:
        svd_solver = trial.suggest_categorical(
            "pca_solver", ReducerTuners.SOLVER_CHOICES
        )

        params = {
            "n_components": trial.suggest_int("pca_components", 16, 128, log=True),
            "whiten": trial.suggest_categorical("pca_whiten", [True, False]),
            "svd_solver": svd_solver,
            "random_state": 42,
        }

        if svd_solver == "randomized":
            params["iterated_power"] = trial.suggest_int("pca_iterated_power", 2, 8)
            params["power_iteration_normalizer"] = trial.suggest_categorical(
                "pca_power_iteration_normalizer", ReducerTuners.NORMALIZER_CHOICES
            )

        return params

    @staticmethod
    def tune_none(trial: optuna.Trial) -> dict[str, Any]:
        return {}

    REGISTRY = {
        "pca": tune_pca,
    }


class ClassifierTuners:
    @staticmethod
    def tune_svm(trial: optuna.Trial) -> dict[str, Any]:
        kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf", "poly"])

        params = {
            "C": trial.suggest_float("svm_c", 0.001, 50, log=True),
            "kernel": kernel,
            "class_weight": trial.suggest_categorical(
                "svm_class_weight", [None, "balanced"]
            ),
            "shrinking": trial.suggest_categorical("svm_shrinking", [True]),
            "random_state": 42,
            "max_iter": 1500,
        }

        if kernel in ["rbf", "poly"]:
            params["gamma"] = trial.suggest_float("svm_gamma", 1e-5, 1e-1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("svm_degree", 2, 5)

        return params

    @staticmethod
    def tune_random_forest(trial: optuna.Trial) -> dict[str, Any]:
        bootstrap = trial.suggest_categorical("rf_bootstrap", [True, False])

        params = {
            "n_estimators": trial.suggest_int("rf_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "rf_max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": bootstrap,
            "criterion": trial.suggest_categorical(
                "rf_criterion", ["gini", "entropy", "log_loss"]
            ),
            "class_weight": trial.suggest_categorical(
                "rf_class_weight", [None, "balanced", "balanced_subsample"]
            ),
            "random_state": 42,
            "n_jobs": -1,
        }

        if bootstrap:
            params["max_samples"] = trial.suggest_float("rf_max_samples", 0.5, 1.0)

        return params

    @staticmethod
    def tune_knn(trial: optuna.Trial) -> dict[str, Any]:
        algorithm = trial.suggest_categorical(
            "knn_algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        )

        if algorithm in ["kd_tree", "ball_tree"]:
            metric = "minkowski"
        else:
            metric = trial.suggest_categorical(
                "knn_metric_brute", ["minkowski", "cosine"]
            )

        return {
            "n_neighbors": trial.suggest_int("knn_neighbors", 3, 21, step=2),
            "weights": trial.suggest_categorical(
                "knn_weights", ["uniform", "distance"]
            ),
            "p": trial.suggest_categorical("knn_p", [1, 2]),
            "metric": metric,
            "leaf_size": trial.suggest_int("knn_leaf_size", 10, 50, step=5),
            "algorithm": algorithm,
            "n_jobs": -1,
        }

    REGISTRY = {
        "svm": tune_svm,
        "random_forest": tune_random_forest,
        "knn": tune_knn,
    }
