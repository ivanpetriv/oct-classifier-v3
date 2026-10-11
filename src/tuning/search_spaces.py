from typing import Any, ClassVar

import optuna


class PreprocessingTuners:
    @staticmethod
    def tune_corner_artifact_remover(
        trial: optuna.Trial,
        prefix: str = "corner_artifact",
    ) -> dict[str, Any]:
        return {
            "min_threshold": trial.suggest_int(
                f"{prefix}_min_threshold",
                200,
                255,
            ),
        }

    @staticmethod
    def tune_resizer(
        trial: optuna.Trial,
        prefix: str = "resizer",
    ) -> dict[str, Any]:
        width = trial.suggest_int(
            f"{prefix}_target_width",
            64,
            512,
            log=True,
        )

        height = trial.suggest_int(
            f"{prefix}_target_height",
            64,
            512,
            log=True,
        )

        return {
            "target_size": (width, height),
        }

    @staticmethod
    def tune_dwt(
        trial: optuna.Trial,
        prefix: str = "dwt",
    ) -> dict[str, Any]:
        return {
            "wavelet": trial.suggest_categorical(
                f"{prefix}_wavelet",
                ["haar", "db2", "db4", "sym2", "sym4"],
            ),
            "mode": trial.suggest_categorical(
                f"{prefix}_mode",
                ["soft", "hard"],
            ),
            "wavelet_levels": trial.suggest_int(
                f"{prefix}_wavelet_levels",
                1,
                4,
            ),
            "method": trial.suggest_categorical(
                f"{prefix}_method",
                ["BayesShrink", "VisuShrink"],
            ),
            "rescale_sigma": trial.suggest_categorical(
                f"{prefix}_rescale_sigma",
                [True, False],
            ),
            "channel_axis": None,
        }

    @staticmethod
    def tune_rpe_roi_aligner(trial: optuna.Trial, prefix: str = "rpe_roi"):
        return {
            "roi_above": trial.suggest_int(
                f"{prefix}_roi_above",
                50,
                250,
            ),
            "roi_below": trial.suggest_int(
                f"{prefix}_roi_below",
                10,
                100,
            ),
            "blur_sigma": trial.suggest_int(
                f"{prefix}_blur_sigma",
                0,
                10,
            ),
            "max_jump": trial.suggest_int(
                f"{prefix}_max_jump",
                low=1,
                high=10,
            ),
            "poly_degree": trial.suggest_int(
                f"{prefix}_poly_degree",
                2,
                8,
            ),
        }

    @staticmethod
    def tune_rpe_aligner(
        trial: optuna.Trial,
        prefix: str = "rpe",
    ) -> dict[str, Any]:
        return {
            "blur_sigma": trial.suggest_int(
                f"{prefix}_blur_sigma",
                1,
                5,
            ),
            "poly_degree": trial.suggest_int(
                f"{prefix}_poly_degree",
                2,
                5,
            ),
            "max_jump": trial.suggest_int(
                f"{prefix}_max_jump",
                2,
                10,
            ),
        }

    REGISTRY: ClassVar = {
        "dwt_denoiser": tune_dwt,
        "rpe_aligner": tune_rpe_aligner,
        "rpe_roi_aligner": tune_rpe_roi_aligner,
        "resizer": tune_resizer,
        "corner_artifact_remover": tune_corner_artifact_remover,
    }


class FeatureExtractorTuners:
    @staticmethod
    def tune_hog(
        trial: optuna.Trial,
        prefix: str = "hog",
    ) -> dict[str, Any]:
        cell_size = trial.suggest_categorical(
            f"{prefix}_cell_size",
            [8, 16, 32],
        )

        block_size = trial.suggest_categorical(
            f"{prefix}_cells_per_block",
            [1, 2, 3],
        )

        return {
            "orientations": trial.suggest_int(
                f"{prefix}_orientations",
                6,
                high=20,
            ),
            "pixels_per_cell": (
                cell_size,
                cell_size,
            ),
            "cells_per_block": (
                block_size,
                block_size,
            ),
            "transform_sqrt": trial.suggest_categorical(
                f"{prefix}_transform_sqrt",
                [True, False],
            ),
            "block_norm": trial.suggest_categorical(
                f"{prefix}_block_norm",
                ["L1", "L1-sqrt", "L2", "L2-Hys"],
            ),
        }

    @staticmethod
    def tune_lbp(
        trial: optuna.Trial,
        prefix: str = "lbp",
    ) -> dict[str, Any]:
        return {
            "n_points": trial.suggest_int(
                f"{prefix}_n_points",
                8,
                24,
                step=8,
            ),
            "radius": trial.suggest_int(
                f"{prefix}_radius",
                1,
                4,
            ),
            "method": trial.suggest_categorical(
                f"{prefix}_method",
                ["uniform", "nri_uniform"],
            ),
            "norm": trial.suggest_categorical(
                f"{prefix}_norm",
                ["l1", "l2"],
            ),
            "padding_threshold": trial.suggest_int(
                f"{prefix}_padding_threshold",
                0,
                50,
            ),
        }

    @staticmethod
    def tune_cdch(
        trial: optuna.Trial,
        prefix: str = "cdch",
    ) -> dict[str, Any]:
        return {
            "angle_step": trial.suggest_int(
                f"{prefix}_angle_step",
                4,
                45,
            ),
            "bins": trial.suggest_int(
                f"{prefix}_bins",
                32,
                256,
                step=32,
            ),
            "pixel_range": (
                trial.suggest_int(
                    f"{prefix}_min_val",
                    0,
                    50,
                ),
                trial.suggest_int(
                    f"{prefix}_max_val",
                    200,
                    255,
                ),
            ),
            "center": (
                trial.suggest_float(
                    f"{prefix}_center_x",
                    0.1,
                    0.9,
                ),
                trial.suggest_float(
                    f"{prefix}_center_y",
                    0.1,
                    0.9,
                ),
            ),
        }

    @staticmethod
    def tune_ldch(
        trial: optuna.Trial,
        prefix: str = "ldch",
    ) -> dict[str, Any]:
        return {
            "sections": trial.suggest_categorical(
                f"{prefix}_sections",
                [32, 64, 128],
            ),
            "bins": trial.suggest_int(
                f"{prefix}_bins",
                64,
                256,
                step=64,
            ),
            "pixel_range": (
                trial.suggest_int(
                    f"{prefix}_min_val",
                    0,
                    50,
                ),
                trial.suggest_int(
                    f"{prefix}_max_val",
                    200,
                    255,
                ),
            ),
            "layout": trial.suggest_categorical(
                f"{prefix}_layout",
                ["row", "col"],
            ),
        }

    @staticmethod
    def tune_rdch(
        trial: optuna.Trial,
        prefix: str = "rdch",
    ) -> dict[str, Any]:
        return {
            "ring_step": trial.suggest_int(
                f"{prefix}_ring_step",
                4,
                128,
            ),
            "bins": trial.suggest_int(
                f"{prefix}_bins",
                64,
                256,
                step=64,
            ),
            "pixel_range": (
                trial.suggest_int(
                    f"{prefix}_min_val",
                    0,
                    50,
                ),
                trial.suggest_int(
                    f"{prefix}_max_val",
                    200,
                    255,
                ),
            ),
            "center": (
                trial.suggest_float(
                    f"{prefix}_center_x",
                    0.1,
                    0.9,
                ),
                trial.suggest_float(
                    f"{prefix}_center_y",
                    0.1,
                    0.9,
                ),
            ),
        }

    @staticmethod
    def tune_rcldch(
        trial: optuna.Trial,
        prefix: str = "rcldch",
    ) -> dict[str, Any]:
        return {
            "bins": trial.suggest_int(
                f"{prefix}_bins",
                64,
                256,
                step=64,
            ),
            "sections": trial.suggest_categorical(
                f"{prefix}_sections",
                [32, 64, 128],
            ),
            "pixel_range": (
                trial.suggest_int(
                    f"{prefix}_min_val",
                    0,
                    50,
                ),
                trial.suggest_int(
                    f"{prefix}_max_val",
                    200,
                    255,
                ),
            ),
        }

    REGISTRY: ClassVar = {
        "hog": tune_hog,
        "lbp": tune_lbp,
        "cdch": tune_cdch,
        "ldch": tune_ldch,
        "rdch": tune_rdch,
        "rcldch": tune_rcldch,
    }


class ReducerTuners:
    @staticmethod
    def tune_pca(
        trial: optuna.Trial,
        prefix: str = "pca",
    ) -> dict[str, Any]:
        solver = trial.suggest_categorical(
            f"{prefix}_solver",
            ["randomized"],
        )

        return {
            "n_components": trial.suggest_int(
                f"{prefix}_components",
                50,
                500,
                log=True,
            ),
            "whiten": trial.suggest_categorical(
                f"{prefix}_whiten",
                [True, False],
            ),
            "svd_solver": solver,
            "random_state": 42,
            "iterated_power": trial.suggest_int(
                f"{prefix}_iterated_power",
                2,
                8,
            ),
            "power_iteration_normalizer": trial.suggest_categorical(
                f"{prefix}_power_iteration_normalizer",
                ["auto", "QR", "LU", "none"],
            ),
        }

    REGISTRY: ClassVar = {
        "pca": tune_pca,
    }


class ClassifierTuners:
    @staticmethod
    def tune_svm(
        trial: optuna.Trial,
        prefix: str = "svm",
    ) -> dict[str, Any]:
        kernel = trial.suggest_categorical(
            f"{prefix}_kernel",
            ["rbf", "poly"],
        )

        params = {
            "kernel": kernel,
            "C": trial.suggest_float(
                f"{prefix}_c",
                0.1,
                100,
                log=True,
            ),
            "class_weight": trial.suggest_categorical(
                f"{prefix}_class_weight",
                ["balanced"],
            ),
            "shrinking": trial.suggest_categorical(
                f"{prefix}_shrinking",
                [True, False],
            ),
            "random_state": 42,
            "max_iter": 2000,
        }

        if kernel in ("rbf", "poly"):
            params["gamma"] = trial.suggest_float(
                f"{prefix}_gamma",
                1e-5,
                1e-1,
                log=True,
            )

        if kernel == "poly":
            params["degree"] = trial.suggest_int(
                f"{prefix}_degree",
                2,
                5,
            )

        return params

    @staticmethod
    def tune_random_forest(
        trial: optuna.Trial,
        prefix: str = "rf",
    ) -> dict[str, Any]:
        bootstrap = trial.suggest_categorical(
            f"{prefix}_bootstrap",
            [True, False],
        )

        params = {
            "n_estimators": trial.suggest_int(
                f"{prefix}_estimators",
                50,
                500,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                f"{prefix}_max_depth",
                5,
                50,
            ),
            "min_samples_split": trial.suggest_int(
                f"{prefix}_min_samples_split",
                2,
                10,
            ),
            "min_samples_leaf": trial.suggest_int(
                f"{prefix}_min_samples_leaf",
                1,
                5,
            ),
            "max_features": trial.suggest_categorical(
                f"{prefix}_max_features",
                ["sqrt", "log2", None],
            ),
            "bootstrap": bootstrap,
            "criterion": trial.suggest_categorical(
                f"{prefix}_criterion",
                ["gini", "entropy", "log_loss"],
            ),
            "class_weight": trial.suggest_categorical(
                f"{prefix}_class_weight",
                [None, "balanced", "balanced_subsample"],
            ),
            "random_state": 42,
            "n_jobs": -1,
        }

        if bootstrap:
            params["max_samples"] = trial.suggest_float(
                f"{prefix}_max_samples",
                0.5,
                1.0,
            )

        return params

    @staticmethod
    def tune_knn(
        trial: optuna.Trial,
        prefix: str = "knn",
    ) -> dict[str, Any]:
        algorithm = trial.suggest_categorical(
            f"{prefix}_algorithm",
            ["auto", "ball_tree", "kd_tree", "brute"],
        )

        if algorithm in ("kd_tree", "ball_tree"):
            metric = "minkowski"
        else:
            metric = trial.suggest_categorical(
                f"{prefix}_metric",
                ["minkowski", "cosine"],
            )

        return {
            "n_neighbors": trial.suggest_int(
                f"{prefix}_neighbors",
                3,
                21,
                step=2,
            ),
            "weights": trial.suggest_categorical(
                f"{prefix}_weights",
                ["uniform", "distance"],
            ),
            "p": trial.suggest_categorical(
                f"{prefix}_p",
                [1, 2],
            ),
            "metric": metric,
            "leaf_size": trial.suggest_int(
                f"{prefix}_leaf_size",
                10,
                50,
                step=5,
            ),
            "algorithm": algorithm,
            "n_jobs": -1,
        }

    @staticmethod
    def tune_logistic_regression(
        trial: optuna.Trial,
        prefix: str = "lr",
    ) -> dict[str, Any]:
        return {
            "C": trial.suggest_float(
                f"{prefix}_c",
                0.01,
                100,
                log=True,
            ),
            "class_weight": trial.suggest_categorical(
                f"{prefix}_class_weight",
                ["balanced", None],
            ),
            "solver": trial.suggest_categorical(
                f"{prefix}_solver",
                ["saga", "lbfgs"],
            ),
            "max_iter": 5000,
            "random_state": 42,
        }

    REGISTRY: ClassVar = {
        "svm": tune_svm,
        "random_forest": tune_random_forest,
        "knn": tune_knn,
        "logistic_regression": tune_logistic_regression,
    }


class PreprocessingDefaults:
    """Recommended default configurations for preprocessing"""

    ACTIVE = ["corner_artifact_remover", "dwt_denoiser", "rpe_roi_aligner", "resizer"]

    DEFAULTS = {
        "corner_artifact_remover": {
            "min_threshold": 230,
        },
        "resizer": {
            "target_size": (256, 256),
        },
        "dwt_denoiser": {
            "wavelet": "db4",
            "mode": "soft",
            "wavelet_levels": 2,
            "method": "BayesShrink",
            "rescale_sigma": True,
        },
        "rpe_aligner": {
            "blur_sigma": 2,
            "poly_degree": 3,
            "max_jump": 5,
        },
        "rpe_roi_aligner": {
            "roi_below": 60,
            "roi_above": 180,
            "blur_sigma": 3,
            "max_jump": 3,
            "poly_degree": 3,
        },
    }


class FeatureExtractorDefaults:
    """Recommended default configurations for feature extraction"""

    ACTIVE = [
        # {
        #     "cdch": {
        #         "angle_step": 15,
        #         "bins": 128,
        #         "pixel_range": (25, 225),
        #         "center": (0.5, 0.5),
        #     },
        # },
        {
            "ldch": {
                "sections": 32,
                "layout": "row",
                "pixel_range": (17, 253),
                "bins": 128,
            }
        },
        # {
        #     "ldch": {
        #         "sections": 64,
        #         "layout": "col",
        #         "pixel_range": (14, 219),
        #         "bins": 192,
        #     }
        # },
        {
            "hog": {
                "orientations": 20,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (2, 2),
                "transform_sqrt": True,
                "block_norm": "L2-Hys",
            }
        },
        # {
        #     "lbp": {
        #         "n_points": 8,
        #         "radius": 1,
        #         "method": "uniform",
        #         "norm": "l2",
        #         "padding_threshold": 20,
        #     },
        # },
    ]

    DEFAULTS = {
        "hog": {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "transform_sqrt": True,
            "block_norm": "L2-Hys",
        },
        "lbp": {
            "n_points": 8,
            "radius": 1,
            "method": "uniform",
            "norm": "l2",
            "padding_threshold": 20,
        },
        "cdch": {
            "angle_step": 15,
            "bins": 128,
            "pixel_range": (25, 225),
            "center": (0.5, 0.5),
        },
        "rdch": {
            "ring_step": 16,
            "bins": 128,
            "pixel_range": (25, 225),
            "center": (0.5, 0.5),
        },
        "rcldch": {
            "bins": 64,
            "sections": 64,
            "pixel_range": (25, 225),
        },
    }


class ReducerDefaults:
    """Recommended default configurations for dimensionality reduction"""

    ACTIVE = "pca"

    DEFAULTS = {
        "pca": {
            "n_components": 54,
            "whiten": False,
            "svd_solver": "randomized",
            "iterated_power": 4,
            "power_iteration_normalizer": "QR",
        },
    }


class ClassifierDefaults:
    """Recommended default configurations for classifiers"""

    # ACTIVE = "random_forest"
    # ACTIVE = "knn"
    # ACTIVE = "logistic_regression"
    ACTIVE = "svm"

    DEFAULTS = {
        "svm": {
            "kernel": "rbf",
            "C": 6.766305087349444,
            "class_weight": "balanced",
            "shrinking": True,
            "gamma": 1.0340550782763686e-05,
            "degree": 5,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "max_samples": 0.8,
            "criterion": "gini",
            "class_weight": "balanced",
        },
        "knn": {
            "n_neighbors": 5,
            "weights": "distance",
            "p": 2,
            "metric": "minkowski",
            "leaf_size": 30,
            "algorithm": "auto",
        },
        "logistic_regression": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
        },
    }


def generate_optuna_seed() -> dict[str, Any]:
    """
    Translates nested dictionaries and tuples into the flat structure
    expected by Optuna's trial.suggest_* calls.
    """
    seed = {}

    # Preprocessing
    for i, step_name in enumerate(PreprocessingDefaults.ACTIVE):
        prefix = f"{step_name}_{i}"
        params = PreprocessingDefaults.DEFAULTS.get(step_name, {})
        for key, value in params.items():
            if key == "target_size":
                seed[f"{prefix}_target_width"] = value[0]
                seed[f"{prefix}_target_height"] = value[1]
            else:
                seed[f"{prefix}_{key}"] = value

    # Feature Extractors
    for i, step_dict in enumerate(FeatureExtractorDefaults.ACTIVE):
        for ext_name, params in step_dict.items():
            prefix = f"{ext_name}_{i}"
            for key, value in params.items():
                if key == "pixels_per_cell":
                    seed[f"{prefix}_cell_size"] = value[0]
                elif key == "cells_per_block":
                    seed[f"{prefix}_cells_per_block"] = value[0]
                elif key == "pixel_range":
                    seed[f"{prefix}_min_val"] = value[0]
                    seed[f"{prefix}_max_val"] = value[1]
                elif key == "center":
                    seed[f"{prefix}_center_x"] = value[0]
                    seed[f"{prefix}_center_y"] = value[1]
                else:
                    seed[f"{prefix}_{key}"] = value

    # Dimensionality Reduction
    red_name = ReducerDefaults.ACTIVE
    if red_name and red_name != "none":
        prefix = red_name
        params = ReducerDefaults.DEFAULTS.get(red_name, {})
        for key, value in params.items():
            if key == "svd_solver":
                seed[f"{prefix}_solver"] = value
            elif key == "n_components":
                seed[f"{prefix}_components"] = value
            else:
                seed[f"{prefix}_{key}"] = value

    # Classifier
    clf_name = ClassifierDefaults.ACTIVE
    if clf_name:
        prefix = clf_name
        params = ClassifierDefaults.DEFAULTS.get(clf_name, {})
        for key, value in params.items():
            if key == "n_estimators":
                seed[f"{prefix}_estimators"] = value
            elif key == "n_neighbors":
                seed[f"{prefix}_neighbors"] = value
            elif key == "C":
                seed[f"{prefix}_c"] = value
            else:
                seed[f"{prefix}_{key}"] = value

    return seed


if __name__ == "__main__":
    print("Preprocessing Defaults:")
    for key, val in PreprocessingDefaults.DEFAULTS.items():
        print(f"  {key}: {val}")

    print("\nFeature Extractor Defaults:")
    for key, val in FeatureExtractorDefaults.DEFAULTS.items():
        print(f"  {key}: {val}")

    print("\nReducer Defaults:")
    for key, val in ReducerDefaults.DEFAULTS.items():
        print(f"  {key}: {val}")

    print("\nClassifier Defaults:")
    for key, val in ClassifierDefaults.DEFAULTS.items():
        print(f"  {key}: {val}")
