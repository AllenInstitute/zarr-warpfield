//! Byte layout of one encoded chunk (identical to `zarr_warpfield.format`).
//!
//! ```text
//! offset  size  field
//! 0       4     magic  b"WFLD"
//! 4       1     format version (2, or 3 with a tight ladder; 1 is still read)
//! 5       1     rank R of the canonical array (batch axis + predicted axes)
//! 6       1     code coder: 0 zigzag LEB128 varints, 1 canonical Huffman (v2)
//! 7       1     reserved, zero
//! 8       8*R   canonical shape, little-endian uint64
//! ...     2*8   level schedule alpha, beta, little-endian float64 (v2 only)
//! ...     4*4   byte lengths of the four sections (five in v3), little-endian uint32
//! ...     3*8   counts: quantization codes, tight indices, verbatim indices (uint64)
//! ...           section 0: quantization codes (coder above), zstd frame
//!               section 1: tight-class sample indices, delta + LEB128, zstd frame
//!               section 2: verbatim sample indices, delta + LEB128, zstd frame
//!               section 3: verbatim values, raw little-endian data type, zstd frame
//!               section 4 (v3): tier of every tight index, one byte each in
//!                          index order (0 = first tight bound), zstd frame
//! ```
//!
//! Version 3 is written only when some tight sample uses a tier past the first.

use crate::error::{Result, format};
use crate::pass::Schedule;

pub const MAGIC: &[u8; 4] = b"WFLD";
pub const VERSION: u8 = 2;
pub const VERSION_LADDER: u8 = 3;
pub const CODER_VARINT: u8 = 0;
pub const CODER_HUFFMAN: u8 = 1;

#[inline(always)]
pub fn zigzag(q: i64) -> u64 {
    ((q << 1) ^ (q >> 63)) as u64
}

#[inline(always)]
pub fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

#[inline(always)]
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Decode exactly `count` LEB128 varints that fill `buf`.
pub fn read_varints(buf: &[u8], count: usize, mut f: impl FnMut(u64)) -> Result<()> {
    let mut n = 0usize;
    let mut v = 0u64;
    let mut shift = 0u32;
    for &b in buf {
        if shift >= 64 {
            return format("varint too long");
        }
        v |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            if n == count {
                return format(format!("more than {count} varints in a section"));
            }
            f(v);
            n += 1;
            v = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
    if shift != 0 {
        return format("truncated varint");
    }
    if n != count {
        return format(format!("expected {count} varints, found {n}"));
    }
    Ok(())
}

fn pack_indices(idx: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(idx.len() * 2);
    let mut prev = 0usize;
    for &i in idx {
        put_varint(&mut out, (i - prev) as u64);
        prev = i;
    }
    out
}

fn unpack_indices(buf: &[u8], count: usize) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(count);
    let mut acc = 0u64;
    read_varints(buf, count, |d| {
        acc = acc.wrapping_add(d);
        out.push(acc as usize);
    })?;
    Ok(out)
}

pub(crate) struct Chunk {
    pub shape: Vec<usize>,
    pub schedule: Schedule,
    pub codes: Vec<i64>,
    pub tight: Vec<usize>,
    /// Tier of each tight sample (all zero before version 3).
    pub tiers: Vec<u8>,
    pub verbatim: Vec<usize>,
    pub values: Vec<u8>, // raw little-endian verbatim values
}

pub(crate) fn varint_codes(codes: &[i64]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(codes.len() + codes.len() / 4);
    for &q in codes {
        put_varint(&mut raw, zigzag(q));
    }
    raw
}

/// Section 0 both ways (varints and Huffman, each then zstd); the smaller
/// wins. With `pick`, both are compressed at that (cheaper) level to choose,
/// and only the winner again at `level`.
fn code_section(codes: &[i64], level: i32, pick: Option<i32>) -> Result<(u8, Vec<u8>)> {
    let raw_v = varint_codes(codes);
    let raw_h = crate::huffman::encode(codes);
    let p = pick.filter(|&p| p < level).unwrap_or(level);
    let v = crate::frames::compress(&raw_v, p)?;
    let h = crate::frames::compress(&raw_h, p)?;
    let (coder, raw, best) = if h.len() < v.len() {
        (CODER_HUFFMAN, &raw_h, h)
    } else {
        (CODER_VARINT, &raw_v, v)
    };
    Ok((
        coder,
        if p < level {
            crate::frames::compress(raw, level)?
        } else {
            best
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write(
    shape: &[usize],
    schedule: Schedule,
    codes: &[i64],
    tight: &[usize],
    tiers: &[u8],
    verbatim: &[usize],
    values: &[u8],
    level: i32,
    pick: Option<i32>,
) -> Result<Vec<u8>> {
    let (coder, sec0) = code_section(codes, level, pick)?;
    let mut sections = vec![
        sec0,
        crate::frames::compress(&pack_indices(tight), level)?,
        crate::frames::compress(&pack_indices(verbatim), level)?,
        crate::frames::compress(values, level)?,
    ];
    let version = if tiers.iter().any(|&t| t != 0) {
        sections.push(crate::frames::compress(tiers, level)?);
        VERSION_LADDER
    } else {
        VERSION
    };
    let total: usize = sections.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(8 + 8 * shape.len() + 16 + 20 + 24 + total);
    out.extend_from_slice(MAGIC);
    out.push(version);
    out.push(shape.len() as u8);
    out.extend_from_slice(&[coder, 0]);
    for &s in shape {
        out.extend_from_slice(&(s as u64).to_le_bytes());
    }
    out.extend_from_slice(&schedule.alpha.to_le_bytes());
    out.extend_from_slice(&schedule.beta.to_le_bytes());
    for s in &sections {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for n in [codes.len(), tight.len(), verbatim.len()] {
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    for s in &sections {
        out.extend_from_slice(s);
    }
    Ok(out)
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.o + n > self.b.len() {
            return format("truncated chunk");
        }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

pub(crate) fn read(buf: &[u8], value_size: usize) -> Result<Chunk> {
    let mut r = Reader { b: buf, o: 0 };
    if r.take(4)? != MAGIC {
        return format("not a warpfield chunk (bad magic)");
    }
    let head = r.take(4)?;
    let version = head[0];
    if !(1..=3).contains(&version) {
        return format(format!("unsupported warpfield format version {version}"));
    }
    let rank = head[1] as usize;
    let coder = head[2];
    if (version == 1 && coder != CODER_VARINT) || coder > CODER_HUFFMAN {
        return format(format!("unknown code coder {coder}"));
    }
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(r.u64()? as usize);
    }
    let schedule = if version == 1 {
        Schedule::UNIFORM
    } else {
        let alpha = f64::from_le_bytes(r.take(8)?.try_into().unwrap());
        let beta = f64::from_le_bytes(r.take(8)?.try_into().unwrap());
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // rejects NaN too
        if !(alpha >= 1.0 && beta >= 1.0 && alpha.is_finite() && beta.is_finite()) {
            return format("level schedule alpha and beta must be finite and >= 1");
        }
        Schedule { alpha, beta }
    };
    let mut lens = vec![0usize; if version == VERSION_LADDER { 5 } else { 4 }];
    for l in &mut lens {
        *l = r.u32()? as usize;
    }
    let n_codes = r.u64()? as usize;
    let n_tight = r.u64()? as usize;
    let n_verb = r.u64()? as usize;
    let n: usize = shape.iter().product();
    if n_codes > n || n_tight > n || n_verb > n {
        return format("section counts exceed the chunk size");
    }
    let mut raw = Vec::with_capacity(lens.len());
    for &l in &lens {
        raw.push(crate::frames::decompress(r.take(l)?)?);
    }
    let codes = if coder == CODER_HUFFMAN {
        crate::huffman::decode(&raw[0], n_codes)?
    } else {
        let mut codes = Vec::with_capacity(n_codes);
        read_varints(&raw[0], n_codes, |z| codes.push(unzigzag(z)))?;
        codes
    };
    let tight = unpack_indices(&raw[1], n_tight)?;
    let verbatim = unpack_indices(&raw[2], n_verb)?;
    if tight.iter().chain(&verbatim).any(|&i| i >= n) {
        return format("sample index out of range");
    }
    let values = std::mem::take(&mut raw[3]);
    if values.len() != n_verb * value_size {
        return format("verbatim value count mismatch");
    }
    let tiers = if lens.len() == 5 {
        let t = std::mem::take(&mut raw[4]);
        if t.len() != n_tight {
            return format("tight tier count mismatch");
        }
        t
    } else {
        vec![0u8; n_tight]
    };
    Ok(Chunk {
        shape,
        schedule,
        codes,
        tight,
        tiers,
        verbatim,
        values,
    })
}
