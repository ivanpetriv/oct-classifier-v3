use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn extract_rdch<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    ring_step: f64,
    center_x: Option<f64>,
    center_y: Option<f64>,
    min_val: u8,
    max_val: u8,
    bins: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = image.as_array();
    let (h, w) = (array.shape()[0], array.shape()[1]);

    // 1. Determine Absolute Center (Casted to f64 for distance precision)
    let (cx, cy) = if let (Some(x), Some(y)) = (center_x, center_y) {
        (x * w as f64, y * h as f64)
    } else {
        let (mut sum_x, mut sum_y, mut count) = (0usize, 0usize, 0usize);
        for ((y, x), &p) in array.indexed_iter() {
            if p >= min_val && p <= max_val {
                sum_x += x;
                sum_y += y;
                count += 1;
            }
        }
        if count > 0 {
            (sum_x as f64 / count as f64, sum_y as f64 / count as f64)
        } else {
            (w as f64 / 2.0, h as f64 / 2.0)
        }
    };

    // 2. Calculate max radius and required rings
    let corners = [
        (0.0, 0.0),
        (w as f64, 0.0),
        (0.0, h as f64),
        (w as f64, h as f64),
    ];
    let max_radius = corners
        .iter()
        .map(|&(kx, ky)| ((kx - cx).powi(2) + (ky - cy).powi(2)).sqrt())
        .fold(0.0, f64::max);

    let num_rings = ((max_radius / ring_step).ceil() as usize).max(1);
    let range = (max_val as usize).saturating_sub(min_val as usize) + 1;

    // 3. Single-pass binning into a flat atomic-friendly buffer
    // To parallelize the histogram build, we use a 1D vector representing (rings * bins)
    let mut ring_counts = vec![0usize; num_rings * bins];
    let mut ring_totals = vec![0usize; num_rings];

    for ((y, x), &p) in array.indexed_iter() {
        if p >= min_val && p <= max_val {
            let dist = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            let ring_idx = ((dist / ring_step) as usize).min(num_rings - 1);

            let bin_idx = (((p - min_val) as usize * bins) / range).min(bins - 1);

            ring_counts[ring_idx * bins + bin_idx] += 1;
            ring_totals[ring_idx] += 1;
        }
    }

    // 4. Parallelize normalization and cumulative sum calculation
    let hists: Vec<Vec<f64>> = (0..num_rings)
        .into_par_iter()
        .map(|r_idx| {
            let mut normalized = vec![0.0; bins];
            let total = ring_totals[r_idx];

            if total > 0 {
                let mut running_sum = 0usize;
                let offset = r_idx * bins;
                for b in 0..bins {
                    running_sum += ring_counts[offset + b];
                    normalized[b] = running_sum as f64 / total as f64;
                }
            }
            normalized
        })
        .collect();

    // 5. Construct Output (Returning raw matrix: shape [num_rings, bins])
    // If you need the rotation for visualization, keep the nested loop;
    // otherwise, this is the most efficient representation.
    let mut out_array = Array2::<f64>::zeros((num_rings, bins));
    for (i, ring_data) in hists.into_iter().enumerate() {
        for (j, val) in ring_data.into_iter().enumerate() {
            out_array[[i, j]] = val;
        }
    }

    Ok(out_array.into_pyarray(py))
}
