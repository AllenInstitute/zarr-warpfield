//! Canonical Huffman coding of the quantization codes (format version 2,
//! code coder 1), the entropy stage of SZ3 (Di and Cappello, IPDPS 2016;
//! Zhao et al., ICDE 2021), followed by zstd as SZ3 does.
//!
//! ```text
//! u16      S: alphabet size; symbol s < S - 1 is zigzag(q) = s, symbol S - 1
//!          is the escape (only when some zigzag(q) >= ESCAPE)
//! S bytes  code length of each symbol (0 = unused, at most MAX_LEN)
//! u64      byte length B of the bit stream
//! B bytes  bit stream: canonical codes, most significant bit first
//! rest     zigzag LEB128 varints of the escaped codes, in order
//! ```
//!
//! Canonical codes: symbols sorted by (length, symbol), codes assigned in that
//! order starting from 0, shifted left at each length step (as in DEFLATE).

use crate::error::{Result, format};
use crate::format::{put_varint, read_varints, unzigzag, zigzag};

/// Symbols zigzag(q) >= ESCAPE are sent as the escape symbol plus a varint.
pub const ESCAPE: u64 = 4095;
pub const MAX_LEN: u8 = 24;
const LUT_BITS: u32 = 12;

/// Code lengths of an optimal Huffman code for `freq`, limited to MAX_LEN by
/// halving the frequencies and rebuilding.
fn lengths(freq: &[u64]) -> Vec<u8> {
    let mut f: Vec<u64> = freq.to_vec();
    loop {
        let len = unlimited(&f);
        if len.iter().all(|&l| l <= MAX_LEN) {
            return len;
        }
        for x in f.iter_mut().filter(|x| **x > 0) {
            *x = (*x).div_ceil(2);
        }
    }
}

/// Two-queue Huffman construction; ties prefer leaves, then lower symbols.
fn unlimited(freq: &[u64]) -> Vec<u8> {
    let mut leaves: Vec<(u64, usize)> = freq
        .iter()
        .enumerate()
        .filter(|(_, f)| **f > 0)
        .map(|(s, &f)| (f, s))
        .collect();
    let mut len = vec![0u8; freq.len()];
    if leaves.len() == 1 {
        len[leaves[0].1] = 1;
        return len;
    }
    leaves.sort_unstable();
    let nl = leaves.len();
    // nodes 0..nl are leaves (in sorted order), nl.. internal
    let mut parent = vec![usize::MAX; 2 * nl - 1];
    let mut weight: Vec<u64> = leaves.iter().map(|l| l.0).collect();
    let (mut li, mut ii) = (0usize, nl);
    let pick = |weight: &Vec<u64>, li: &mut usize, ii: &mut usize| -> usize {
        let take_leaf = *li < nl && (*ii >= weight.len() || weight[*li] <= weight[*ii]);
        if take_leaf {
            *li += 1;
            *li - 1
        } else {
            *ii += 1;
            *ii - 1
        }
    };
    for _ in 0..nl - 1 {
        let a = pick(&weight, &mut li, &mut ii);
        let b = pick(&weight, &mut li, &mut ii);
        let n = weight.len();
        weight.push(weight[a] + weight[b]);
        parent[a] = n;
        parent[b] = n;
    }
    // depth of internal nodes, from the root (last) down
    let total = 2 * nl - 1;
    let mut depth = vec![0u32; total];
    for n in (0..total - 1).rev() {
        depth[n] = depth[parent[n]] + 1;
    }
    for (k, &(_, s)) in leaves.iter().enumerate() {
        len[s] = depth[k].min(255) as u8;
    }
    len
}

/// Canonical codes for `len` (0 where unused).
fn canonical(len: &[u8]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..len.len()).filter(|&s| len[s] > 0).collect();
    order.sort_by_key(|&s| (len[s], s));
    let mut code = vec![0u32; len.len()];
    let mut c = 0u32;
    let mut prev = 0u8;
    for (k, &s) in order.iter().enumerate() {
        if k > 0 {
            c += 1;
        }
        c <<= len[s] - prev;
        prev = len[s];
        code[s] = c;
    }
    code
}

#[inline(always)]
fn symbol(z: u64) -> usize {
    z.min(ESCAPE) as usize
}

/// Huffman-code `codes` (section 0 payload, before zstd).
pub fn encode(codes: &[i64]) -> Vec<u8> {
    let mut freq = vec![0u64; ESCAPE as usize + 1];
    let mut max_sym = 0usize;
    for &q in codes {
        let s = symbol(zigzag(q));
        freq[s] += 1;
        max_sym = max_sym.max(s);
    }
    if codes.is_empty() {
        let mut out = vec![0u8; 2];
        out.extend_from_slice(&0u64.to_le_bytes());
        return out;
    }
    let nsym = max_sym + 1;
    freq.truncate(nsym);
    let len = lengths(&freq);
    let code = canonical(&len);
    let mut out = Vec::with_capacity(2 + nsym + 8 + codes.len() / 8 + 16);
    out.extend_from_slice(&(nsym as u16).to_le_bytes());
    out.extend_from_slice(&len);
    let mut bits = Vec::with_capacity(codes.len() / 8 + 8);
    let mut acc = 0u64;
    let mut nacc = 0u32;
    let mut escapes = Vec::new();
    for &q in codes {
        let z = zigzag(q);
        let s = symbol(z);
        if z >= ESCAPE {
            put_varint(&mut escapes, z);
        }
        let l = len[s] as u32;
        acc = (acc << l) | code[s] as u64;
        nacc += l;
        while nacc >= 8 {
            nacc -= 8;
            bits.push((acc >> nacc) as u8);
        }
        acc &= (1u64 << nacc) - 1;
    }
    if nacc > 0 {
        bits.push((acc << (8 - nacc)) as u8);
    }
    out.extend_from_slice(&(bits.len() as u64).to_le_bytes());
    out.extend_from_slice(&bits);
    out.extend_from_slice(&escapes);
    out
}

/// Inverse of `encode`, for exactly `count` codes.
pub fn decode(buf: &[u8], count: usize) -> Result<Vec<i64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let take = |o: usize, n: usize| -> Result<&[u8]> {
        buf.get(o..o + n)
            .ok_or_else(|| crate::Error::Format("truncated Huffman section".into()))
    };
    let nsym = u16::from_le_bytes(take(0, 2)?.try_into().unwrap()) as usize;
    let len = take(2, nsym)?;
    if len.iter().any(|&l| l > MAX_LEN) {
        return format("Huffman code length exceeds the limit");
    }
    let nbytes = u64::from_le_bytes(take(2 + nsym, 8)?.try_into().unwrap()) as usize;
    let o = 2 + nsym + 8;
    let bits = take(o, nbytes)?;
    let esc_buf = &buf[o + nbytes..];

    // canonical decoding tables: first code and first index per length
    let mut order: Vec<usize> = (0..nsym).filter(|&s| len[s] > 0).collect();
    if order.is_empty() {
        return format("empty Huffman code");
    }
    order.sort_by_key(|&s| (len[s], s));
    let mut count_l = [0u32; MAX_LEN as usize + 1];
    for &s in &order {
        count_l[len[s] as usize] += 1;
    }
    // Kraft check: a complete or single-symbol code
    let kraft: u64 = (1..=MAX_LEN as usize)
        .map(|l| (count_l[l] as u64) << (MAX_LEN as usize - l))
        .sum();
    if kraft > 1u64 << MAX_LEN {
        return format("over-subscribed Huffman code");
    }
    let mut first_code = [0u32; MAX_LEN as usize + 2];
    let mut first_idx = [0u32; MAX_LEN as usize + 2];
    let mut c = 0u32;
    let mut idx = 0u32;
    for l in 1..=MAX_LEN as usize {
        first_code[l] = c;
        first_idx[l] = idx;
        c = (c + count_l[l]) << 1;
        idx += count_l[l];
    }
    // lookup table for codes up to LUT_BITS: (symbol, length)
    let code = canonical(len);
    let mut lut = vec![(0u16, 0u8); 1 << LUT_BITS];
    for &s in &order {
        let l = len[s] as u32;
        if l <= LUT_BITS {
            let base = (code[s] << (LUT_BITS - l)) as usize;
            for e in &mut lut[base..base + (1 << (LUT_BITS - l))] {
                *e = (s as u16, l as u8);
            }
        }
    }
    let escape = if nsym > 0 && (nsym - 1) as u64 == ESCAPE {
        Some(nsym - 1)
    } else {
        None
    };
    let mut escaped: Vec<u64> = Vec::new();
    read_all_varints(esc_buf, &mut escaped)?;
    let mut esc_i = 0usize;

    let mut out = Vec::with_capacity(count);
    let mut window = 0u64; // left-aligned bits
    let mut nwin = 0u32;
    let mut pos = 0usize;
    let total_bits = nbytes as u64 * 8;
    let mut used = 0u64;
    for _ in 0..count {
        while nwin <= 56 {
            let b = if pos < nbytes { bits[pos] } else { 0 };
            pos += 1;
            window |= (b as u64) << (56 - nwin);
            nwin += 8;
        }
        let top = (window >> (64 - LUT_BITS)) as usize;
        let (mut s, mut l) = (lut[top].0 as usize, lut[top].1 as u32);
        if l == 0 {
            // slow path: codes longer than LUT_BITS
            let mut found = false;
            for ll in LUT_BITS + 1..=MAX_LEN as u32 {
                let v = (window >> (64 - ll)) as u32;
                let n = count_l[ll as usize];
                if n > 0 && v >= first_code[ll as usize] && v - first_code[ll as usize] < n {
                    s = order[(first_idx[ll as usize] + v - first_code[ll as usize]) as usize];
                    l = ll;
                    found = true;
                    break;
                }
            }
            if !found {
                return format("invalid Huffman code");
            }
        }
        used += l as u64;
        if used > total_bits {
            return format("truncated Huffman bit stream");
        }
        window <<= l;
        nwin -= l;
        if Some(s) == escape {
            let Some(&z) = escaped.get(esc_i) else {
                return format("missing escaped code");
            };
            esc_i += 1;
            out.push(unzigzag(z));
        } else {
            out.push(unzigzag(s as u64));
        }
    }
    if esc_i != escaped.len() {
        return format("unused escaped codes");
    }
    Ok(out)
}

fn read_all_varints(buf: &[u8], out: &mut Vec<u64>) -> Result<()> {
    let n = buf.iter().filter(|&&b| b < 0x80).count();
    read_varints(buf, n, |v| out.push(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut v: Vec<i64> = (0..10000).map(|i| ((i * 7919) % 13) as i64 - 6).collect();
        v.extend([0; 5000]);
        v.extend([5000, -100000, 1 << 40, 3]);
        assert_eq!(decode(&encode(&v), v.len()).unwrap(), v);
        assert_eq!(decode(&encode(&[]), 0).unwrap(), Vec::<i64>::new());
        let z = vec![0i64; 1000];
        assert_eq!(decode(&encode(&z), z.len()).unwrap(), z);
        // skewed enough to need length limiting
        let mut f = Vec::new();
        for k in 0..40 {
            f.extend(std::iter::repeat_n(k as i64, 1usize << (k / 2).min(20)));
        }
        assert_eq!(decode(&encode(&f), f.len()).unwrap(), f);
    }
}
