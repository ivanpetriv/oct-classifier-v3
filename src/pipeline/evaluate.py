from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    report: dict

    @property
    def macro_f1(self) -> float:
        return self.report["macro avg"]["f1-score"]


class Evaluator:
    @staticmethod
    def evaluate(
        model,
        X_test,
        y_test,
        classes: list[str],
    ) -> EvaluationResult:

        preds = model.predict(X_test)

        report = classification_report(
            y_test,
            preds,
            target_names=classes,
            zero_division=0,
            output_dict=True,
        )

        return EvaluationResult(
            y_true=y_test,
            y_pred=preds,
            report=report,
        )
