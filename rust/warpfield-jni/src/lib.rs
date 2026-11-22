//! JNI bindings for the warpfield codec: the native half of `org.warpfield.Warpfield`
//! (the Java module in `java/`).
//!
//! Java arrays are accessed in place with `GetPrimitiveArrayCritical`, and direct
//! `ByteBuffer`s by address, so a decode copies nothing between the JVM and Rust.
//! Errors and panics become `org.warpfield.WarpfieldException`; no Java exception
//! is raised while a critical region is held.
//!
//! Bounds checks (offsets, lengths, shape) are done here as well as in Java, so
//! a bad call from Java can never read or write out of bounds.

use std::panic::{AssertUnwindSafe, catch_unwind};

use jni::JNIEnv;
use jni::objects::{
    JByteArray, JByteBuffer, JClass, JDoubleArray, JFloatArray, JIntArray, JPrimitiveArray,
    JString, ReleaseMode, TypeArray,
};
use jni::sys::{jbyteArray, jdouble, jint, jstring};
use warpfield::{Config, Element, EncodeOptions};

const EXCEPTION: &str = "org/warpfield/WarpfieldException";

type Res<T> = Result<T, String>;

fn jerr(e: jni::errors::Error) -> String {
    e.to_string()
}

/// Run `f`, turning an `Err` or a panic into a `WarpfieldException`
/// (unless a Java exception is already pending) and returning `fallback`.
fn guard<'local, R>(
    env: &mut JNIEnv<'local>,
    fallback: R,
    f: impl FnOnce(&mut JNIEnv<'local>) -> Res<R>,
) -> R {
    let result = catch_unwind(AssertUnwindSafe(|| f(env)));
    let msg = match result {
        Ok(Ok(r)) => return r,
        Ok(Err(m)) => m,
        Err(p) => {
            let what = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            format!("warpfield panicked: {what}")
        }
    };
    if !env.exception_check().unwrap_or(true) {
        let _ = env.throw_new(EXCEPTION, msg);
    }
    fallback
}

fn config(env: &mut JNIEnv, cfg: &JString) -> Res<Config> {
    let s: String = env.get_string(cfg).map_err(jerr)?.into();
    Config::from_json(&s).map_err(|e| e.to_string())
}

fn shape(env: &mut JNIEnv, shape: &JIntArray) -> Res<(Vec<usize>, usize)> {
    let n = env.get_array_length(shape).map_err(jerr)? as usize;
    let mut v = vec![0i32; n];
    env.get_int_array_region(shape, 0, &mut v).map_err(jerr)?;
    let mut size = 1usize;
    let mut out = Vec::with_capacity(n);
    for &s in &v {
        if s < 0 {
            return Err(format!("negative extent in shape {v:?}"));
        }
        size = size
            .checked_mul(s as usize)
            .ok_or_else(|| format!("shape {v:?} is too large"))?;
        out.push(s as usize);
    }
    Ok((out, size))
}

/// `[off, off + len)` within an array of `total` elements.
fn range(off: jint, len: usize, total: usize, what: &str) -> Res<std::ops::Range<usize>> {
    if off < 0 {
        return Err(format!("negative {what} offset {off}"));
    }
    let off = off as usize;
    match off.checked_add(len) {
        Some(end) if end <= total => Ok(off..end),
        _ => Err(format!(
            "{what}: {len} elements from offset {off} do not fit in {total}"
        )),
    }
}

/// Decode `src[src_off .. src_off + src_len]` into `dst[dst_off ..]`.
#[allow(clippy::too_many_arguments)]
fn decode_array<T: Element + TypeArray>(
    env: &mut JNIEnv,
    cfg: &JString,
    src: &JByteArray,
    src_off: jint,
    src_len: jint,
    shp: &JIntArray,
    dst: &JPrimitiveArray<T>,
    dst_off: jint,
) -> Res<()> {
    let cfg = config(env, cfg)?;
    let (shape, n) = shape(env, shp)?;
    let src_total = env.get_array_length(src).map_err(jerr)? as usize;
    let dst_total = env.get_array_length(dst).map_err(jerr)? as usize;
    if src_len < 0 {
        return Err(format!("negative source length {src_len}"));
    }
    let src_range = range(src_off, src_len as usize, src_total, "source")?;
    let dst_range = range(dst_off, n, dst_total, "destination")?;
    // Two critical regions at once are allowed by JNI; each guard borrows its
    // own handle to the same environment. Nothing below calls back into Java.
    let mut env2 = unsafe { env.unsafe_clone() };
    let s =
        unsafe { env.get_array_elements_critical(src, ReleaseMode::NoCopyBack) }.map_err(jerr)?;
    let mut d =
        unsafe { env2.get_array_elements_critical(dst, ReleaseMode::CopyBack) }.map_err(jerr)?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) };
    warpfield::decode_into(&bytes[src_range], &shape, &cfg, &mut d[dst_range])
        .map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn encode_array<'local, T: Element + TypeArray>(
    env: &mut JNIEnv<'local>,
    cfg: &JString,
    data: &JPrimitiveArray<T>,
    off: jint,
    shp: &JIntArray,
    zstd_level: jint,
    alpha: f64,
    beta: f64,
) -> Res<JByteArray<'local>> {
    let cfg = config(env, cfg)?;
    let (shape, n) = shape(env, shp)?;
    let total = env.get_array_length(data).map_err(jerr)? as usize;
    let r = range(off, n, total, "data")?;
    let opts = EncodeOptions {
        zstd_level,
        classes: false,
        // (0, 0) asks the encoder to tune the level schedule per chunk
        schedule: (alpha != 0.0 || beta != 0.0).then_some(warpfield::Schedule { alpha, beta }),
    };
    // Encoding takes far longer than decoding (zstd level 19), so copy the
    // input out rather than hold a critical region (which can stall the GC).
    let x: Vec<T> = {
        let el = unsafe { env.get_array_elements(data, ReleaseMode::NoCopyBack) }.map_err(jerr)?;
        el[r].to_vec()
    };
    let enc = warpfield::encode(&x, &shape, &cfg, &opts).map_err(|e| e.to_string())?;
    env.byte_array_from_slice(&enc.bytes).map_err(jerr)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_warpfield_Warpfield_nVersion<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
) -> jstring {
    guard(&mut env, std::ptr::null_mut(), |env| {
        Ok(env
            .new_string(env!("CARGO_PKG_VERSION"))
            .map_err(jerr)?
            .into_raw())
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_warpfield_Warpfield_nDecodeFloat32<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    cfg: JString<'l>,
    src: JByteArray<'l>,
    src_off: jint,
    src_len: jint,
    shape: JIntArray<'l>,
    dst: JFloatArray<'l>,
    dst_off: jint,
) {
    guard(&mut env, (), |env| {
        decode_array::<f32>(env, &cfg, &src, src_off, src_len, &shape, &dst, dst_off)
    })
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_warpfield_Warpfield_nDecodeFloat64<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    cfg: JString<'l>,
    src: JByteArray<'l>,
    src_off: jint,
    src_len: jint,
    shape: JIntArray<'l>,
    dst: JDoubleArray<'l>,
    dst_off: jint,
) {
    guard(&mut env, (), |env| {
        decode_array::<f64>(env, &cfg, &src, src_off, src_len, &shape, &dst, dst_off)
    })
}

/// Decode between direct buffers: `src` (a direct ByteBuffer) bytes
/// `[src_pos, src_pos + src_len)`, into `dst` (a direct FloatBuffer or
/// DoubleBuffer, native byte order) from element `dst_pos`. `dtype`: 0 float32,
/// 1 float64 (the element type of `dst`). JNI reports a direct buffer's
/// capacity in its own elements, so `dst` positions are in elements too.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_warpfield_Warpfield_nDecodeDirect<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    cfg: JString<'l>,
    src: JByteBuffer<'l>,
    src_pos: jint,
    src_len: jint,
    shape_arr: JIntArray<'l>,
    dtype: jint,
    // a FloatBuffer or DoubleBuffer; JNI's direct-buffer calls take any Buffer
    dst: JByteBuffer<'l>,
    dst_pos: jint,
) {
    guard(&mut env, (), |env| {
        let cfg = config(env, &cfg)?;
        let (shape, n) = shape(env, &shape_arr)?;
        let width = match dtype {
            0 => 4,
            1 => 8,
            _ => return Err(format!("unknown dtype code {dtype}")),
        };
        let src_ptr = env.get_direct_buffer_address(&src).map_err(jerr)?;
        let src_cap = env.get_direct_buffer_capacity(&src).map_err(jerr)?;
        let dst_ptr = env.get_direct_buffer_address(&dst).map_err(jerr)?;
        let dst_cap = env.get_direct_buffer_capacity(&dst).map_err(jerr)?;
        if src_len < 0 {
            return Err(format!("negative source length {src_len}"));
        }
        let sr = range(src_pos, src_len as usize, src_cap, "source")?;
        let dr = range(dst_pos, n, dst_cap, "destination")?;
        let out = unsafe { dst_ptr.add(dr.start * width) };
        if !(out as usize).is_multiple_of(width) {
            return Err(format!(
                "destination must be {width}-byte aligned (a view of a direct ByteBuffer \
                 at a byte offset that is a multiple of {width})"
            ));
        }
        let buf = unsafe { std::slice::from_raw_parts(src_ptr.add(sr.start), sr.len()) };
        let r = if width == 4 {
            let o = unsafe { std::slice::from_raw_parts_mut(out as *mut f32, n) };
            warpfield::decode_into(buf, &shape, &cfg, o)
        } else {
            let o = unsafe { std::slice::from_raw_parts_mut(out as *mut f64, n) };
            warpfield::decode_into(buf, &shape, &cfg, o)
        };
        r.map_err(|e| e.to_string())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_warpfield_Warpfield_nEncodeFloat32<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    cfg: JString<'l>,
    data: JFloatArray<'l>,
    off: jint,
    shape: JIntArray<'l>,
    zstd_level: jint,
    alpha: jdouble,
    beta: jdouble,
) -> jbyteArray {
    guard(&mut env, std::ptr::null_mut(), |env| {
        Ok(encode_array::<f32>(env, &cfg, &data, off, &shape, zstd_level, alpha, beta)?.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_warpfield_Warpfield_nEncodeFloat64<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    cfg: JString<'l>,
    data: JDoubleArray<'l>,
    off: jint,
    shape: JIntArray<'l>,
    zstd_level: jint,
    alpha: jdouble,
    beta: jdouble,
) -> jbyteArray {
    guard(&mut env, std::ptr::null_mut(), |env| {
        Ok(encode_array::<f64>(env, &cfg, &data, off, &shape, zstd_level, alpha, beta)?.into_raw())
    })
}
