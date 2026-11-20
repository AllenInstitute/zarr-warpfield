//! Python module `zarr_warpfield_rs`: the Rust warpfield encoder/decoder.
//!
//! Low-level on purpose; `zarr_warpfield` wraps it. The configuration is
//! passed as the JSON text of the codec's `configuration` object. The GIL is
//! released while coding, so zarr's thread pool decodes chunks in parallel.

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use warpfield::{Config, Element, EncodeOptions, Error};

pyo3::create_exception!(zarr_warpfield_rs, SpecError, PyValueError);
pyo3::create_exception!(zarr_warpfield_rs, FormatError, PyValueError);

fn err(e: Error) -> PyErr {
    match e {
        Error::Spec(m) => SpecError::new_err(m),
        Error::Format(m) => FormatError::new_err(m),
    }
}

fn config(json: &str) -> PyResult<Config> {
    Config::from_json(json).map_err(err)
}

type EncodeResult<'py> = (
    Bound<'py, PyBytes>,
    Option<Bound<'py, PyArrayDyn<u8>>>,
    usize,
    usize,
    usize,
    usize,
);

fn encode_typed<'py, T: Element + numpy::Element>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, T>,
    cfg: &Config,
    opts: &EncodeOptions,
) -> PyResult<EncodeResult<'py>> {
    let shape = arr.shape().to_vec();
    let data = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    let enc = py
        .detach(|| warpfield::encode(data, &shape, cfg, opts))
        .map_err(err)?;
    let classes = match enc.classes {
        Some(c) => Some(c.into_pyarray(py).reshape(shape)?),
        None => None,
    };
    Ok((
        PyBytes::new(py, &enc.bytes),
        classes,
        enc.n_codes,
        enc.n_tight,
        enc.n_verbatim,
        enc.jacobian_rounds,
    ))
}

/// encode(arr, config_json, *, classes=False, zstd_level=19)
///     -> (bytes, classes | None, n_codes, n_tight, n_verbatim, jacobian_rounds)
///
/// `arr` is a C-contiguous float32 or float64 numpy array holding one chunk.
#[pyfunction]
#[pyo3(signature = (arr, config, *, classes = false, zstd_level = warpfield::DEFAULT_ZSTD_LEVEL))]
fn encode<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    config: &str,
    classes: bool,
    zstd_level: i32,
) -> PyResult<EncodeResult<'py>> {
    let cfg = self::config(config)?;
    let opts = EncodeOptions {
        zstd_level,
        classes,
    };
    if let Ok(a) = arr.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        encode_typed(py, a, &cfg, &opts)
    } else if let Ok(a) = arr.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        encode_typed(py, a, &cfg, &opts)
    } else {
        Err(SpecError::new_err(
            "data type must be a float32 or float64 numpy array",
        ))
    }
}

fn decode_typed<'py, T: Element + numpy::Element + Default>(
    py: Python<'py>,
    buf: &[u8],
    shape: Vec<usize>,
    cfg: &Config,
) -> PyResult<Bound<'py, PyAny>> {
    let out: Vec<T> = py
        .detach(|| warpfield::decode(buf, &shape, cfg))
        .map_err(err)?;
    Ok(out.into_pyarray(py).reshape(shape)?.into_any())
}

/// decode(buf, shape, dtype, config_json) -> numpy array of `shape` and `dtype`
#[pyfunction]
fn decode<'py>(
    py: Python<'py>,
    buf: &[u8],
    shape: Vec<usize>,
    dtype: &str,
    config: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cfg = self::config(config)?;
    match dtype {
        "float32" | "<f4" | "f4" => decode_typed::<f32>(py, buf, shape, &cfg),
        "float64" | "<f8" | "f8" => decode_typed::<f64>(py, buf, shape, &cfg),
        _ => Err(SpecError::new_err(format!(
            "data type must be float32 or float64, not {dtype}"
        ))),
    }
}

/// decode_into(buf, out, config_json): decode into an existing C-contiguous array.
#[pyfunction]
fn decode_into<'py>(
    py: Python<'py>,
    buf: &[u8],
    out: &Bound<'py, PyAny>,
    config: &str,
) -> PyResult<()> {
    let cfg = self::config(config)?;
    fn run<T: Element + numpy::Element>(
        py: Python<'_>,
        buf: &[u8],
        a: &Bound<'_, PyArrayDyn<T>>,
        cfg: &Config,
    ) -> PyResult<()> {
        let mut rw = a.try_readwrite()?;
        let shape = rw.shape().to_vec();
        let slice = rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output array must be C-contiguous"))?;
        py.detach(|| warpfield::decode_into(buf, &shape, cfg, slice))
            .map_err(err)
    }
    if let Ok(a) = out.cast::<PyArrayDyn<f32>>() {
        run(py, buf, a, &cfg)
    } else if let Ok(a) = out.cast::<PyArrayDyn<f64>>() {
        run(py, buf, a, &cfg)
    } else {
        Err(SpecError::new_err(
            "output must be a float32 or float64 numpy array",
        ))
    }
}

/// validate(config_json): raise SpecError if the configuration is invalid.
#[pyfunction]
fn validate(config: &str) -> PyResult<()> {
    self::config(config).map(|_| ())
}

#[pymodule]
fn zarr_warpfield_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add("SpecError", m.py().get_type::<SpecError>())?;
    m.add("FormatError", m.py().get_type::<FormatError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_ZSTD_LEVEL", warpfield::DEFAULT_ZSTD_LEVEL)?;
    Ok(())
}
