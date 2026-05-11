import optuna


class FeatureExtractorTuners:
    @staticmethod
    def tune_hog(trial: optuna.Trial, params):
        params.orientations = trial.suggest_int("hog_orientations", 6, 12)
        cell_size = trial.suggest_categorical("hog_cell_size", [16, 32, 64])
        params.pixels_per_cell = [cell_size, cell_size]

        # Added: cells_per_block and transform_sqrt
        block_size = trial.suggest_categorical("hog_cells_per_block", [1, 2])
        params.cells_per_block = [block_size, block_size]
        params.transform_sqrt = trial.suggest_categorical(
            "hog_transform_sqrt", [True, False]
        )

    @staticmethod
    def tune_lbp(trial: optuna.Trial, params):
        params.n_points = trial.suggest_int("lbp_n_points", 8, 24, step=8)
        params.radius = trial.suggest_int("lbp_radius", 1, 3)
        # Added: method (uniform is critical for texture)
        params.method = trial.suggest_categorical(
            "lbp_method", ["default", "uniform", "nri_uniform"]
        )

    @staticmethod
    def tune_cdch(trial: optuna.Trial, params):
        params.angle_step = trial.suggest_categorical("cdch_angle_step", [2, 4, 8])
        params.bins = trial.suggest_int("cdch_bins", 64, 256, step=64)

    @staticmethod
    def tune_dch(trial: optuna.Trial, params):
        params.sections = trial.suggest_categorical("dch_sections", [32, 64, 128])
        params.bins = trial.suggest_int("dch_bins", 64, 256, step=64)

    @staticmethod
    def tune_rdch(trial: optuna.Trial, params):
        params.rings = trial.suggest_categorical("rdch_rings", [32, 64, 128])
        params.bins = trial.suggest_int("rdch_bins", 64, 256, step=64)

    @staticmethod
    def tune_rcldch(trial: optuna.Trial, params):
        params.bins = trial.suggest_int("rcldch_bins", 64, 256, step=64)
        params.sections = trial.suggest_categorical("rcldch_sections", [32, 64, 128])
        params.pixel_range = [
            trial.suggest_int("rcldch_min_val", 0, 50),
            trial.suggest_int("rcldch_max_val", 200, 255),
        ]

    # Registry attached directly to the class for easy import/access
    REGISTRY = {
        "hog": tune_hog,
        "lbp": tune_lbp,
        "cdch": tune_cdch,
        "dch": tune_dch,
        "rdch": tune_rdch,
        "rcldch": tune_rcldch,
    }


class ReducerTuners:
    @staticmethod
    def tune_pca(trial: optuna.Trial, params):
        params.n_components = trial.suggest_int("pca_components", 50, 500, step=50)
        params.whiten = trial.suggest_categorical("pca_whiten", [True, False])
        # Added: svd_solver
        params.svd_solver = trial.suggest_categorical(
            "pca_solver", ["auto", "randomized"]
        )

    @staticmethod
    def tune_none(trial: optuna.Trial, params):
        pass

    REGISTRY = {
        "pca": tune_pca,
        "none": tune_none,
    }


class ClassifierTuners:
    @staticmethod
    def tune_svm(trial: optuna.Trial, params):
        params.C = trial.suggest_float("svm_c", 0.01, 100.0, log=True)
        params.kernel = trial.suggest_categorical(
            "svm_kernel", ["linear", "rbf", "poly"]
        )

        # Added: class_weight to handle imbalanced datasets
        params.class_weight = trial.suggest_categorical(
            "svm_class_weight", [None, "balanced"]
        )

        if params.kernel in ["rbf", "poly"]:
            params.gamma = trial.suggest_categorical("svm_gamma", ["scale", "auto"])

        # Added: degree (Only matters if kernel is poly)
        if params.kernel == "poly":
            params.degree = trial.suggest_int("svm_degree", 2, 4)

    @staticmethod
    def tune_random_forest(trial: optuna.Trial, params):
        params.n_estimators = trial.suggest_int("rf_estimators", 50, 300, step=50)
        params.max_depth = trial.suggest_categorical("rf_max_depth", [None, 10, 20, 30])
        params.min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)

        # Added: min_samples_leaf, max_features, and class_weight
        params.min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 4)
        params.max_features = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", None]
        )
        params.class_weight = trial.suggest_categorical(
            "rf_class_weight", [None, "balanced", "balanced_subsample"]
        )

    @staticmethod
    def tune_knn(trial: optuna.Trial, params):
        params.n_neighbors = trial.suggest_int("knn_neighbors", 3, 15, step=2)
        params.weights = trial.suggest_categorical(
            "knn_weights", ["uniform", "distance"]
        )

        # Added: p (1 = Manhattan distance, 2 = Euclidean distance)
        params.p = trial.suggest_categorical("knn_p", [1, 2])

    REGISTRY = {
        "svm": tune_svm,
        "random_forest": tune_random_forest,
        "knn": tune_knn,
    }
