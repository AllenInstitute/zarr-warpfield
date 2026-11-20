//! Byte layout of one encoded chunk (identical to `zarr_warpfield.format`).
//!
//! ```text
//! offset  size  field
//! 0       4     magic  b"WFLD"
//! 4       1     format version (1)
//! 5       1     rank R of the canonical array (batch axis + predicted axes)
//! 6       2     reserved, zero
//! 8       8*R   canonical shape, little-endian uint64
//! ...     4*4   byte lengths of the four sections, little-endian uint32
//! ...     3*8   counts: quantization codes, tight indices, verbatim indices (uint64)
//! ...           section 0: quantization codes, zigzag LEB128 varints, zstd frame
//!               section 1: tight-class sample indices, delta + LEB128, zstd frame
//!               section 2: verbatim sample indices, delta + LEB128, zstd frame
//!               section 3: verbatim values, raw little-endian data type, zstd frame
//! ```

use crate::error::{Result, format};

pub const MAGIC: &[u8; 4] = b"WFLD";
pub const VERSION: u8 = 1;

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
    pub codes: Vec<i64>,
    pub tight: Vec<usize>,
    pub verbatim: Vec<usize>,
    pub values: Vec<u8>, // raw little-endian verbatim values
}

pub(crate) fn write(
    shape: &[usize],
    codes: &[i64],
    tight: &[usize],
    verbatim: &[usize],
    values: &[u8],
    level: i32,
) -> Result<Vec<u8>> {
    let mut raw0 = Vec::with_capacity(codes.len() + codes.len() / 4);
    for &q in codes {
        put_varint(&mut raw0, zigzag(q));
    }
    let sections = [
        crate::frames::compress(&raw0, level)?,
        crate::frames::compress(&pack_indices(tight), level)?,
        crate::frames::compress(&pack_indices(verbatim), level)?,
        crate::frames::compress(values, level)?,
    ];
    let total: usize = sections.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(8 + 8 * shape.len() + 16 + 24 + total);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(shape.len() as u8);
    out.extend_from_slice(&[0, 0]);
    for &s in shape {
        out.extend_from_slice(&(s as u64).to_le_bytes());
    }
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
    if head[0] != VERSION {
        return format(format!("unsupported warpfield format version {}", head[0]));
    }
    let rank = head[1] as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(r.u64()? as usize);
    }
    let mut lens = [0usize; 4];
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
    let mut raw = Vec::with_capacity(4);
    for &l in &lens {
        raw.push(crate::frames::decompress(r.take(l)?)?);
    }
    let mut codes = Vec::with_capacity(n_codes);
    read_varints(&raw[0], n_codes, |z| codes.push(unzigzag(z)))?;
    let tight = unpack_indices(&raw[1], n_tight)?;
    let verbatim = unpack_indices(&raw[2], n_verb)?;
    if tight.iter().chain(&verbatim).any(|&i| i >= n) {
        return format("sample index out of range");
    }
    let values = std::mem::take(&mut raw[3]);
    if values.len() != n_verb * value_size {
        return format("verbatim value count mismatch");
    }
    Ok(Chunk {
        shape,
        codes,
        tight,
        verbatim,
        values,
    })
}
