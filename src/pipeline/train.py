import matplotlib.pyplot as plt
import numpy as np

from config_settings.oct_config import OctConfig
from pipeline.pipeline_builder import build_pipeline


class Trainer:
    def __init__(self, config: OctConfig):
        self.config = config
        self.pipeline = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Builds and fits the machine learning pipeline."""
        self.pipeline = build_pipeline(
            self.config,
            True,
            True,
            True,
            True,
        )
        self.pipeline.fit(X_train, y_train)
        self._analyze_pca()

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Generates predictions using the trained pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Cannot predict before training the pipeline.")
        return self.pipeline.predict(X_val)

    def _analyze_pca(self) -> None:
        """Inspects the dimensionality reduction step after fitting."""
        if (
            self.pipeline is None
            or "dimensionality_reduction" not in self.pipeline.named_steps
        ):
            return

        pca_step = self.pipeline.named_steps["dimensionality_reduction"]

        if hasattr(pca_step, "explained_variance_ratio_"):
            cumsum_var = np.cumsum(pca_step.explained_variance_ratio_)

            print("\n=== PCA Variance Analysis ===")
            print(
                f"Variance in {pca_step.n_components} components: {cumsum_var[-1]:.1%}"
            )

            for threshold in [0.8, 0.9, 0.95]:
                n_comp_needed = np.argmax(cumsum_var >= threshold) + 1
                print(f"Components for {threshold:.0%} variance: {n_comp_needed}")

            plt.figure(figsize=(10, 5))
            plt.plot(cumsum_var[:200])
            plt.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
            plt.xlabel("PCA Components")
            plt.ylabel("Cumulative Variance Explained")
            plt.title("PCA Variance Profile")
            plt.legend()
            plt.savefig("pca_variance.png")
            plt.close()
