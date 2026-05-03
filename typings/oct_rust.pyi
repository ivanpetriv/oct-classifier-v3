# typings/oct_rust.pyi
import numpy as np

def extract_ldch(
    image: np.ndarray,
    sections: int,
    is_row_layout: bool,
    min_val: int,
    max_val: int,
    bins: int
) -> np.ndarray:
    """
    Extracts the cumulative directional histogram using PyO3 and Rust Rayon.
    """
    ...

def extract_cdch(
    image: np.ndarray,
    angle_step: int,
    center_x: float | None,
    center_y: float | None,
    min_val: int,
    max_val: int,
    bins: int
) -> np.ndarray:
    """
    Extracts Circular Distributed Cumulative Histograms using Rust Rayon.
    Calculates rays radially and automatically applies a 90-degree counter-clockwise rotation.
    """
    ...

def extract_rdch(
    image: np.ndarray,
    ring_step: float,
    center_x: float | None,
    center_y: float | None,
    min_val: int,
    max_val: int,
    bins: int
) -> np.ndarray:
    """
    Extracts Ring Distributed Cumulative Histograms using Rust Rayon.
    Dynamically calculates the number of rings based on ring_step and image bounds.
    """
    ...