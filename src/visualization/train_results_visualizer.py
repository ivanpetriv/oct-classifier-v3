from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class TrainResultsVisualizer:
    @staticmethod
    def show_confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
    ):
        """
        Generates a comprehensive visual classification report window containing
        a Confusion Matrix and a per-class F1-Score bar chart.
        """
        # 1. Calculate Metrics
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        )

        # Extract F1 scores for the bar chart (ignoring the 'accuracy' and 'macro/weighted avg' keys)
        f1_scores = [report[cls]["f1-score"] for cls in classes]

        # 2. Setup the Figure
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]}
        )
        fig.suptitle("Model Evaluation Report", fontsize=16, fontweight="bold")

        # --- Subplot 1: The Confusion Matrix Heatmap ---
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Number of Samples"},
        )

        axes[0].set_title("Confusion Matrix", fontsize=14, pad=15)
        axes[0].set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("True Label", fontsize=12, fontweight="bold")

        # Rotate x-axis labels if they are long
        plt.setp(
            axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        # --- Subplot 2: Per-Class F1-Scores Bar Chart ---
        bars = axes[1].bar(classes, f1_scores, alpha=0.8)

        # Add the exact numbers on top of the bars
        axes[1].bar_label(bars, fmt="%.2f", padding=3)

        axes[1].set_title("F1-Score per Class", fontsize=14, pad=15)
        axes[1].set_ylim(0, 1.1)  # Set Y limit slightly above 1 to make room for labels
        axes[1].set_ylabel("F1-Score", fontsize=12)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        # 3. Add Overall Accuracy as text
        accuracy = report["accuracy"]
        fig.text(
            0.5,
            0.02,
            f"Overall Model Accuracy: {accuracy * 100:.2f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"
            ),
        )

        plt.tight_layout(
            rect=(0.0, 0.05, 1.0, 0.95)
        )  # Adjust layout to make room for the suptitle and bottom text
        plt.show()
