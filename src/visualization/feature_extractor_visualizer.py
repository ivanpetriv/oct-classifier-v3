import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor
from feature_extraction.extractors.cdch_feature_extractor import CdchFeaturesExtractor
from feature_extraction.extractors.ldch_feature_extractor import LdchFeaturesExtractor
from feature_extraction.extractors.rdch_feature_extractor import RdchFeaturesExtractor


class FeatureExtractorVisualizer:
    """
    Visualizer for geometric feature extraction overlays and resulting heatmaps.
    Can render multiple extractors side-by-side in a single figure.
    """

    def __init__(self, feature_extractors: list[BaseFeaturesExtractor]):
        self.feature_extractors = feature_extractors

    def _visualize_ldch(
        self, extractor: LdchFeaturesExtractor, image: np.ndarray, ax_img: plt.Axes
    ) -> tuple[str, str]:
        image_height, image_width = image.shape[:2]
        title = f"LDCH ({extractor.sections} {extractor.layout}s)"

        if extractor.layout == "row":
            step = image_height / extractor.sections
            for i in range(1, extractor.sections):
                ax_img.axhline(
                    int(i * step), color="red", linewidth=0.5, alpha=0.7, linestyle="--"
                )
            return title, "Row Section Index"

        elif extractor.layout == "col":
            step = image_width / extractor.sections
            for i in range(1, extractor.sections):
                ax_img.axvline(
                    int(i * step), color="red", linewidth=0.5, alpha=0.7, linestyle="--"
                )
            return title, "Column Section Index"

        return title, "Index"

    def _visualize_rdch(
        self, extractor: RdchFeaturesExtractor, image: np.ndarray, ax_img: plt.Axes
    ) -> tuple[str, str]:
        image_height, image_width = image.shape[:2]

        if extractor.center is None:
            extractor.center = (0.5, 0.5)

        abs_center_x = int(extractor.center[0] * image_width)
        abs_center_y = int(extractor.center[1] * image_height)

        ax_img.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)

        corners = np.array(
            [[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]]
        )
        max_radius = np.max(
            np.hypot(corners[:, 0] - abs_center_x, corners[:, 1] - abs_center_y)
        )

        ring_width = extractor.ring_step
        num_rings = int(np.ceil(max_radius / ring_width))

        for r in range(1, num_rings + 1):
            circle = plt.Circle(
                (abs_center_x, abs_center_y),
                r * ring_width,
                color="red",
                fill=False,
                linewidth=0.5,
                alpha=0.7,
                linestyle="--",
            )
            ax_img.add_patch(circle)

        return f"RDCH (Px: [{abs_center_x}, {abs_center_y}])", "Ring Index"

    def _visualize_cdch(
        self, extractor: CdchFeaturesExtractor, image: np.ndarray, ax_img: plt.Axes
    ) -> tuple[str, str]:
        image_height, image_width = image.shape[:2]

        if extractor.center is None:
            extractor.center = (0.5, 0.5)

        abs_center_x = int(extractor.center[0] * image_width)
        abs_center_y = int(extractor.center[1] * image_height)

        ax_img.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)
        max_radius = int(np.hypot(image_width, image_height))
        all_angles = np.arange(0, 360, extractor.angle_step) - 90
        all_radians = np.deg2rad(all_angles)

        for angle in all_radians:
            end_x = abs_center_x + max_radius * np.cos(angle)
            end_y = abs_center_y + max_radius * np.sin(angle)
            ax_img.plot(
                [abs_center_x, end_x],
                [abs_center_y, end_y],
                color="red",
                linewidth=0.5,
                alpha=0.7,
            )

        return f"CDCH (Center: [{abs_center_x}, {abs_center_y}])", "Ray Angle Index"

    def visualize(self, image: np.ndarray, save_path: str | None = None) -> None:
        num_extractors = len(self.feature_extractors)
        if num_extractors == 0:
            print("No feature extractors provided to visualize.")
            return

        image_height, image_width = image.shape[:2]

        # Change to N rows, 2 columns.
        # Increase vertical size dynamically based on the number of extractors.
        fig, axes = plt.subplots(
            nrows=num_extractors,
            ncols=2,
            figsize=(14, 5 * num_extractors),
            squeeze=False,
        )

        for row_idx, extractor in enumerate(self.feature_extractors):
            ax_img = axes[row_idx, 0]
            ax_heat = axes[row_idx, 1]

            features = extractor.extract(image)
            is_ldch = isinstance(extractor, LdchFeaturesExtractor)

            if is_ldch:
                features = features.T

            if isinstance(extractor, CdchFeaturesExtractor):
                features = np.flipud(features)

            ax_img.imshow(image, cmap="gray")
            ax_img.set_xlim(0, image_width)
            ax_img.set_ylim(image_height, 0)
            ax_img.axis("off")

            if isinstance(extractor, CdchFeaturesExtractor):
                title, dynamic_label = self._visualize_cdch(extractor, image, ax_img)
            elif is_ldch:
                title, dynamic_label = self._visualize_ldch(extractor, image, ax_img)
            elif isinstance(extractor, RdchFeaturesExtractor):
                title, dynamic_label = self._visualize_rdch(extractor, image, ax_img)
            else:
                title = type(extractor).__name__
                dynamic_label = "Index"

            ax_img.set_title(title, fontsize=22)

            im = ax_heat.imshow(
                features,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                origin="lower",
            )

            ax_heat.set_title(f"{type(extractor).__name__}", fontsize=22)

            if is_ldch:
                ax_heat.set_ylabel("Intensity Bins", fontsize=18)
                ax_heat.set_xlabel(dynamic_label, fontsize=18)
            else:
                ax_heat.set_ylabel("Intensity Bins", fontsize=18)
                ax_heat.set_xlabel(dynamic_label, fontsize=18)

            ax_heat.tick_params(axis="both", which="major", labelsize=16)

            divider = make_axes_locatable(ax_heat)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(
                "Normalized Frequency", rotation=270, labelpad=20, fontsize=18
            )
            cbar.ax.tick_params(labelsize=16)

        # Ensure subplots don't overlap their labels
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
