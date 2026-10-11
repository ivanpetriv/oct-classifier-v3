import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class TrainResultsVisualizer:
    @staticmethod
    def show_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list[str],
        font_scale: float = 0.9,
    ):
        """
        Generates a comprehensive visual classification report window containing
        a Confusion Matrix and a per-class F1-Score bar chart.

        font_scale multiplies every font size (1.0 = large default).
        """
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        )

        f1_scores = [report[cls]["f1-score"] for cls in classes]

        def s(size: float) -> float:
            return size * font_scale

        with plt.rc_context(
            {
                "font.size": s(22),
                "xtick.labelsize": s(20),
                "ytick.labelsize": s(20),
                "axes.labelsize": s(24),
                "axes.titlesize": s(26),
            }
        ):
            fig, axes = plt.subplots(
                1, 2, figsize=(24, 11), gridspec_kw={"width_ratios": [1.5, 1]}
            )
            fig.suptitle("Model Evaluation Report", fontsize=s(32), fontweight="bold")

            heatmap = sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[0],
                xticklabels=classes,
                yticklabels=classes,
                annot_kws={"size": s(22)},
            )
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=s(20))
            cbar.set_label("Number of Samples", fontsize=s(22))

            axes[0].set_title("Confusion Matrix for HOG", fontsize=s(26), pad=20)
            axes[0].set_xlabel("Predicted Label", fontsize=s(24), fontweight="bold")
            axes[0].set_ylabel("True Label", fontsize=s(24), fontweight="bold")

            plt.setp(
                axes[0].get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            bars = axes[1].bar(classes, f1_scores, alpha=0.8)
            axes[1].bar_label(bars, fmt="%.2f", padding=4, fontsize=s(20))

            axes[1].set_title("F1-Score per Class", fontsize=s(26), pad=20)
            axes[1].set_ylim(0, 1.12)
            axes[1].set_ylabel("F1-Score", fontsize=s(24))
            axes[1].grid(axis="y", linestyle="--", alpha=0.5)
            plt.setp(
                axes[1].get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
            plt.show()
