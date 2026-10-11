use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ndarray::Array2, ndarray::Axis};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn extract_ldch<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
    sections: usize,
    is_row_layout: bool,
    pixel_min: u8,
    pixel_max: u8,
    bins: usize,
) -> Bound<'py, PyArray2<f64>> {
    assert!(sections > 0, "sections must be > 0");
    assert!(bins > 0, "bins must be > 0");

    let array = image.as_array();
    let axis = if is_row_layout { Axis(0) } else { Axis(1) };
    let total_dim = array.len_of(axis);

    let base_size = total_dim / sections;
    let remainder = total_dim % sections;

    let range = (pixel_max as usize).saturating_sub(pixel_min as usize) + 1;

    let hists: Vec<Vec<f64>> = (0..sections)
        .into_par_iter()
        .map(|i| {
            // Determine section boundaries
            let size = if i < remainder {
                base_size + 1
            } else {
                base_size
            };
            let current_offset = if i < remainder {
                i * (base_size + 1)
            } else {
                remainder * (base_size + 1) + (i - remainder) * base_size
            };

            let section = array.slice_axis(axis, (current_offset..current_offset + size).into());

            let mut counts = vec![0usize; bins];
            let mut total_valid = 0usize;

            // Single pass over section pixels
            for &pixel in section.iter() {
                if pixel >= pixel_min && pixel <= pixel_max {
                    let bin_idx = (((pixel - pixel_min) as usize * bins) / range).min(bins - 1);
                    counts[bin_idx] += 1;
                    total_valid += 1;
                }
            }

            let mut normalized = vec![0.0; bins];
            if total_valid > 0 {
                let mut running_sum = 0usize;
                for b in 0..bins {
                    running_sum += counts[b];
                    normalized[b] = running_sum as f64 / total_valid as f64;
                }
            }
            normalized
        })
        .collect();

    let mut out_array = Array2::<f64>::zeros((sections, bins));
    for (i, row_data) in hists.into_iter().enumerate() {
        for (j, val) in row_data.into_iter().enumerate() {
            out_array[[i, j]] = val;
        }
    }

    out_array.into_pyarray(py)
}
