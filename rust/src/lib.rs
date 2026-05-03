mod feature_extractors;

use pyo3::prelude::*;
use feature_extractors::ldch_feature_extractor::extract_ldch;
use feature_extractors::cdch_feature_extractor::extract_cdch;
use feature_extractors::rdch_feature_extractor::extract_rdch;

#[pymodule]
fn oct_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_ldch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_cdch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_rdch, m)?)?;
    Ok(())
}