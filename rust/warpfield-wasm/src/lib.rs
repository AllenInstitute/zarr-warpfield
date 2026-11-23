//! WebAssembly exports for the warpfield codec.
//!
//! A plain C ABI (no wasm-bindgen), so the module can be instantiated with
//! `WebAssembly.instantiate` and no JS glue beyond `js/src/index.ts`. Memory
//! protocol: the host allocates input buffers with `wf_alloc` (8-byte
//! aligned), calls `wf_decode` / `wf_encode`, and frees with `wf_free`. On
//! failure a call returns a non-zero status and the message is available via
//! `wf_error_ptr` / `wf_error_len`. `wf_encode` leaves its output in a result
//! buffer read with `wf_result_ptr` / `wf_result_len`.

use std::alloc::{Layout, alloc, dealloc};
use std::cell::RefCell;

use warpfield::{Config, EncodeOptions, Error};

thread_local! {
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

const ALIGN: usize = 8;

fn set_error(e: impl std::fmt::Display) -> i32 {
    ERROR.with(|s| *s.borrow_mut() = e.to_string());
    1
}

/// Allocate `n` bytes, 8-byte aligned (n = 0 allowed).
#[unsafe(no_mangle)]
pub extern "C" fn wf_alloc(n: usize) -> *mut u8 {
    unsafe { alloc(Layout::from_size_align(n.max(1), ALIGN).unwrap()) }
}

/// Free a buffer from `wf_alloc` of the same size.
///
/// # Safety
/// `p` must come from `wf_alloc(n)` and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wf_free(p: *mut u8, n: usize) {
    unsafe { dealloc(p, Layout::from_size_align(n.max(1), ALIGN).unwrap()) }
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_error_ptr() -> *const u8 {
    ERROR.with(|s| s.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_error_len() -> usize {
    ERROR.with(|s| s.borrow().len())
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn wf_result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

/// Release the result buffer of the last `wf_encode`.
#[unsafe(no_mangle)]
pub extern "C" fn wf_result_clear() {
    RESULT.with(|r| *r.borrow_mut() = Vec::new());
}

unsafe fn parse(cfg_ptr: *const u8, cfg_len: usize) -> Result<Config, Error> {
    let s = unsafe { std::slice::from_raw_parts(cfg_ptr, cfg_len) };
    let s = std::str::from_utf8(s).map_err(|_| Error::Spec("configuration is not UTF-8".into()))?;
    Config::from_json(s)
}

unsafe fn shape_of(p: *const u32, rank: usize) -> Vec<usize> {
    unsafe { std::slice::from_raw_parts(p, rank) }
        .iter()
        .map(|&v| v as usize)
        .collect()
}

/// Decode a chunk into `out` (prod(shape) elements of `dtype`: 0 = float32, 1 = float64).
///
/// # Safety
/// All pointers must reference valid memory of the stated sizes; `out` must be
/// 4- (float32) or 8-byte (float64) aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wf_decode(
    cfg_ptr: *const u8,
    cfg_len: usize,
    buf_ptr: *const u8,
    buf_len: usize,
    shape_ptr: *const u32,
    rank: usize,
    dtype: u32,
    out_ptr: *mut u8,
) -> i32 {
    let run = || -> Result<(), Error> {
        let cfg = unsafe { parse(cfg_ptr, cfg_len) }?;
        let shape = unsafe { shape_of(shape_ptr, rank) };
        let n: usize = shape.iter().product();
        let buf = unsafe { std::slice::from_raw_parts(buf_ptr, buf_len) };
        match dtype {
            0 => {
                let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, n) };
                warpfield::decode_into(buf, &shape, &cfg, out)
            }
            1 => {
                let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f64, n) };
                warpfield::decode_into(buf, &shape, &cfg, out)
            }
            _ => Err(Error::Spec(format!("unknown dtype code {dtype}"))),
        }
    };
    match run() {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

/// Encode a chunk; the bytes are left in the result buffer. `alpha`, `beta`:
/// the level schedule, or (0, 0) to tune it per chunk. `effort`: 0 fast, 1 max.
/// `zstd_level` <= 0 takes the effort's level.
///
/// # Safety
/// As for `wf_decode`; `data_ptr` holds prod(shape) values of `dtype`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wf_encode(
    cfg_ptr: *const u8,
    cfg_len: usize,
    data_ptr: *const u8,
    shape_ptr: *const u32,
    rank: usize,
    dtype: u32,
    zstd_level: i32,
    alpha: f64,
    beta: f64,
    effort: u32,
) -> i32 {
    let run = || -> Result<Vec<u8>, Error> {
        let cfg = unsafe { parse(cfg_ptr, cfg_len) }?;
        let shape = unsafe { shape_of(shape_ptr, rank) };
        let n: usize = shape.iter().product();
        let base = EncodeOptions::with_effort(if effort == 1 {
            warpfield::Effort::Max
        } else {
            warpfield::Effort::Fast
        });
        let opts = EncodeOptions {
            zstd_level: if zstd_level > 0 {
                zstd_level
            } else {
                base.zstd_level
            },
            // (0, 0) asks the encoder to tune the level schedule per chunk
            schedule: (alpha != 0.0 || beta != 0.0).then_some(warpfield::Schedule { alpha, beta }),
            ..base
        };
        let enc = match dtype {
            0 => {
                let d = unsafe { std::slice::from_raw_parts(data_ptr as *const f32, n) };
                warpfield::encode(d, &shape, &cfg, &opts)?
            }
            1 => {
                let d = unsafe { std::slice::from_raw_parts(data_ptr as *const f64, n) };
                warpfield::encode(d, &shape, &cfg, &opts)?
            }
            _ => return Err(Error::Spec(format!("unknown dtype code {dtype}"))),
        };
        Ok(enc.bytes)
    };
    match run() {
        Ok(b) => {
            RESULT.with(|r| *r.borrow_mut() = b);
            0
        }
        Err(e) => set_error(e),
    }
}
