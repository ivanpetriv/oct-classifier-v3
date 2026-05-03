from typing import cast

from PIL import Image
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
    """

    def __init__(self, feature_extractor: BaseFeaturesExtractor):
        self.feature_extractor = feature_extractor

    def _visualize_ldch(self, image: np.ndarray, ax_img: plt.Axes) -> tuple[str, str]:
        """Draws DCH (Linear) grid lines and returns (title, y_label)."""
        feature_extractor = cast(LdchFeaturesExtractor, self.feature_extractor)

        image_height, image_width = image.shape[:2]

        title = f"DCH ({feature_extractor.sections} {feature_extractor.layout}s)"

        if feature_extractor.layout == "row":
            step = image_height / feature_extractor.sections
            for i in range(1, feature_extractor.sections):
                y = int(i * step)
                ax_img.axhline(y, color="red", linewidth=0.5, alpha=0.7, linestyle="--")
            return title, "Row Section Index"

        elif feature_extractor.layout == "col":
            step = image_width / feature_extractor.sections
            for i in range(1, feature_extractor.sections):
                x = int(i * step)
                ax_img.axvline(x, color="red", linewidth=0.5, alpha=0.7, linestyle="--")
            return title, "Column Section Index"

        return title, "Index"

    def _visualize_rdch(self, image: np.ndarray, ax_img: plt.Axes) -> tuple[str, str]:
        """Draws RDCH concentric rings and returns (title, y_label)."""
        feature_extractor = cast(RdchFeaturesExtractor, self.feature_extractor)

        image_height, image_width = image.shape[:2]

        # Calculate absolute center (Handling normalized [0; 1] coordinates)
        if (
            feature_extractor.center_x is not None
            and feature_extractor.center_y is not None
        ):
            abs_center_x = int(feature_extractor.center_x * image_width)
            abs_center_y = int(feature_extractor.center_y * image_height)
        else:
            abs_center_x, abs_center_y = image_width // 2, image_height // 2

        ax_img.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)

        corners = np.array(
            [[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]]
        )
        max_radius = np.max(
            np.hypot(corners[:, 0] - abs_center_x, corners[:, 1] - abs_center_y)
        )

        ring_width = feature_extractor.ring_step
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

        return (
            f"Rings (Px: [{abs_center_x}, {abs_center_y}])",
            "Ring Index (Inner to Outer)",
        )

    def _visualize_cdch(self, image: np.ndarray, ax_img: plt.Axes) -> tuple[str, str]:
        """Draws CDCH rays and returns (title, y_label)."""
        feature_extractor = cast(CdchFeaturesExtractor, self.feature_extractor)

        image_height, image_width = image.shape[:2]

        # Calculate absolute center (Handling normalized [0; 1] coordinates)
        if (
            feature_extractor.center_x is not None
            and feature_extractor.center_y is not None
        ):
            abs_center_x = int(feature_extractor.center_x * image_width)
            abs_center_y = int(feature_extractor.center_y * image_height)
        else:
            abs_center_x, abs_center_y = image_width // 2, image_height // 2

        ax_img.scatter(abs_center_x, abs_center_y, color="lime", s=4, zorder=3)

        max_radius = int(np.hypot(image_width, image_height))
        all_angles = np.arange(0, 360, feature_extractor.angle_step) - 90
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

        return f"Rays (Px: [{abs_center_x}, {abs_center_y}])", "Ray Angle Index"

    def visualize(self, image: np.ndarray, save_path: str | None = None) -> None:
        """
        Executes the feature extractor on the image and plots the overlay alongside the heatmap.
        """
        features = self.feature_extractor.extract(image)
        image_height, image_width = image.shape[:2]

        fig, (ax_img, ax_heat) = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

        # 1. Base Image Setup
        ax_img.imshow(image, cmap="gray")
        ax_img.set_xlim(0, image_width)
        ax_img.set_ylim(image_height, 0)
        ax_img.axis("off")

        # 2. Dynamic Overlay Routing
        if isinstance(self.feature_extractor, CdchFeaturesExtractor):
            title, y_label = self._visualize_cdch(image, ax_img)
        elif isinstance(self.feature_extractor, LdchFeaturesExtractor):
            title, y_label = self._visualize_ldch(image, ax_img)
        elif isinstance(self.feature_extractor, RdchFeaturesExtractor):
            title, y_label = self._visualize_rdch(image, ax_img)
        else:
            title = type(self.feature_extractor).__name__
            y_label = "Index"

        ax_img.set_title(title)

        # 3. Heatmap Setup
        heatmap_h, heatmap_w = features.shape
        target_aspect = (image_height / image_width) * (heatmap_w / heatmap_h)

        im = ax_heat.imshow(
            features, cmap="gray", aspect=target_aspect, interpolation="nearest"
        )
        ax_heat.set_title(f"Extracted Matrix ({type(self.feature_extractor).__name__})")
        ax_heat.set_xlabel("Intensity Bins")
        ax_heat.set_ylabel(y_label)

        divider = make_axes_locatable(ax_heat)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Normalized Frequency", rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    feature_extractor = LdchFeaturesExtractor(256, "col", (0, 255), 256)
    visualizer = FeatureExtractorVisualizer(feature_extractor)
    with Image.open(
        r"/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli-normalized/train/CNV/CNV-163081-45.jpeg"
    ) as file_object:
        visualizer.visualize(np.array(file_object))
