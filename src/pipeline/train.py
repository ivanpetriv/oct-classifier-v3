from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from dataset.oct_dataset import OctDataset
from pipeline.evaluate import EvaluationResult, Evaluator

# --- 1. The Trainer (Handles the ML Lifecycle) ---


class SklearnTrainer:
    """
    Responsible for the actual model lifecycle: tuning, fitting, and predicting.
    It knows HOW to train, but doesn't care WHERE the data came from.
    """

    def __init__(self, pipeline: Pipeline):
        self.model = pipeline
        self.is_trained = False

    def tune_and_fit(self, X_train, y_train, X_val=None, y_val=None, tuner=None):
        # 1. Tune if a tuner is provided
        if tuner is not None:
            if X_val is None or y_val is None:
                raise ValueError("Validation data must be provided if a tuner is used.")

            print("Running hyperparameter tuning...")
            best_params = tuner.optimize(self.model, X_train, y_train, X_val, y_val)
            self.model.set_params(**best_params)

        # 2. Fit the model (with or without new params)
        print("Fitting final model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def save(self, path: str | Path):
        if not self.is_trained:
            print("Warning: Saving an untrained model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)


# --- 2. The Experiment (The Orchestrator) ---


class Experiment:
    """
    Responsible for orchestrating the flow of data to the trainer and evaluator.
    It lazy-loads data to protect system RAM.
    """

    def __init__(self, dataset: OctDataset, classes: list[str]):
        self.dataset = dataset
        self.classes = classes

    def run(self, trainer: SklearnTrainer, tuner=None) -> EvaluationResult:
        # Step 1: Load Train Data
        print("Loading Training Data...")
        X_train, y_train = self.dataset.gather_full_split("train", self.classes)

        # Step 2: Load Val Data ONLY if we are tuning (Saves RAM)
        X_val, y_val = None, None
        if tuner is not None:
            print("Loading Validation Data...")
            X_val, y_val = self.dataset.gather_full_split("val", self.classes)

        # Step 3: Train & Tune
        trainer.tune_and_fit(X_train, y_train, X_val, y_val, tuner)

        # Step 4: Free up Train/Val memory before loading Test data
        del X_train, y_train, X_val, y_val

        # Step 5: Load Test Data & Evaluate
        print("Loading Test Data...")
        X_test, y_test = self.dataset.gather_full_split("test", self.classes)

        print("Evaluating Model...")
        result = Evaluator.evaluate(trainer.model, X_test, y_test, self.classes)

        return result
