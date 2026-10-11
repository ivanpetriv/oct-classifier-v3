use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn extract_cdch<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    angle_step: usize,
    center_x: Option<f64>,
    center_y: Option<f64>,
    min_val: u8,
    max_val: u8,
    bins: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = image.as_array();
    let (h, w) = (array.shape()[0], array.shape()[1]);

    // Center Calculation
    let (cx, cy) = if let (Some(x), Some(y)) = (center_x, center_y) {
        ((x * w as f64).round() as i32, (y * h as f64).round() as i32)
    } else {
        // Fallback to center of mass
        let (mut sum_x, mut sum_y, mut count) = (0usize, 0usize, 0usize);
        for ((y, x), &p) in array.indexed_iter() {
            if p >= min_val && p <= max_val {
                sum_x += x;
                sum_y += y;
                count += 1;
            }
        }
        if count > 0 {
            ((sum_x / count) as i32, (sum_y / count) as i32)
        } else {
            ((w / 2) as i32, (h / 2) as i32)
        }
    };

    let max_radius = (w as f64).hypot(h as f64) as i32;
    let angles_deg: Vec<f64> = (0..360)
        .step_by(angle_step)
        .map(|a| a as f64 - 90.0)
        .collect();
    let num_rays = angles_deg.len();

    let range = (max_val as usize).saturating_sub(min_val as usize) + 1;

    let ray_hists: Vec<Vec<f64>> = angles_deg
        .into_par_iter()
        .map(|angle_deg| {
            let rad = angle_deg.to_radians();
            let (cos_a, sin_a) = (rad.cos(), rad.sin());
            let mut counts = vec![0usize; bins];
            let mut total_valid = 0usize;

            for r in 0..max_radius {
                let px = (cx as f64 + r as f64 * cos_a).round() as i32;
                let py = (cy as f64 + r as f64 * sin_a).round() as i32;

                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    let p = array[[py as usize, px as usize]];
                    if p >= min_val && p <= max_val {
                        let bin_idx = (((p - min_val) as usize * bins) / range).min(bins - 1);
                        counts[bin_idx] += 1;
                        total_valid += 1;
                    }
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

    // Matrix Construction
    let mut out_array = Array2::<f64>::zeros((bins, num_rays));
    for i in 0..num_rays {
        for j in 0..bins {
            out_array[[bins - 1 - j, i]] = ray_hists[i][j];
        }
    }

    Ok(out_array.into_pyarray(py))
}
