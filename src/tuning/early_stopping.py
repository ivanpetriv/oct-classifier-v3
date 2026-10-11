import optuna


class EarlyStoppingCallback:
    """Terminates the Optuna study if the objective fails to improve."""

    def __init__(self, patience: int):
        self.patience = patience
        self.trials_without_improvement = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if study.best_trial.number == trial.number:
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.patience:
            print(
                f"\n[Early Stopping Triggered] No improvement in F1-score for {self.patience} consecutive trials."
            )
            print(
                "The hyperparameter space has converged. Halting study to save compute time."
            )
            study.stop()
