//! zstd frames, from libzstd (`zstd-native`) or pure Rust (`zstd-pure`).

use crate::error::Result;

#[cfg(feature = "zstd-native")]
pub(crate) fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
    // bulk API: the frame records its content size, like python-zstandard
    zstd::bulk::compress(data, level).map_err(|e| crate::Error::Format(format!("zstd: {e}")))
}

#[cfg(feature = "zstd-native")]
pub(crate) fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    zstd::stream::decode_all(data).map_err(|e| crate::Error::Format(format!("zstd: {e}")))
}

#[cfg(all(feature = "zstd-pure", not(feature = "zstd-native")))]
pub(crate) fn compress(data: &[u8], _level: i32) -> Result<Vec<u8>> {
    Ok(ruzstd::encoding::compress_to_vec(
        data,
        ruzstd::encoding::CompressionLevel::Fastest,
    ))
}

#[cfg(all(feature = "zstd-pure", not(feature = "zstd-native")))]
pub(crate) fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut src = data;
    let mut dec = ruzstd::decoding::StreamingDecoder::new(&mut src)
        .map_err(|e| crate::Error::Format(format!("zstd: {e}")))?;
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|e| crate::Error::Format(format!("zstd: {e}")))?;
    Ok(out)
}

#[cfg(not(any(feature = "zstd-native", feature = "zstd-pure")))]
compile_error!("enable one of the features `zstd-native` or `zstd-pure`");
