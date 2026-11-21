"""warpfield vs zfp (fixed-accuracy) on the full ANTs fields.

zfp's `tolerance` is a loose upper bound: its realized error is usually far
below it. So zfp is run at a sweep of tolerances and compared with warpfield at
matched *realized* error, both max and RMS.

Streams the field in k-slabs of one chunk row (64 planes) and keeps the last
plane of the previous slab, so every cell is Jacobian-checked, including cells
that straddle chunks. For each codec setting it reports:

- size and ratio vs raw float32;
- max and RMS error over all samples and components;
- Jacobian violations (cells that drop to det J <= min_det or flip sign);
- seams: the |error jump| between neighbours across a chunk boundary, mean and
  max, and the mean divided by the same inside chunks. warpfield bounds the
  boundary jump by 2 * face_error_bound (eps / 2); zfp codes chunks
  independently with no such bound;
- single-thread encode / decode time.

Reads the lossless zstd store and the warpfield stores written by verify.py
straight from their chunk files, so zarr is not imported (zfpy 1.0.1.post0's
version string breaks numcodecs inside zarr):

    uv run --with "zfpy==1.0.1.post0" python zfp_compare.py OUT

Results go to OUT/zfp_compare.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zfpy
import zstandard

from warpfield import Config, core, jacobian

EPS = ("1um", "2.5um", "6.25um")
ZFP_TOL_MM = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
MIN_DET = 0.1


class Store:
    """A zarr v3 array read from its chunk files (default chunk keys)."""

    def __init__(self, path: Path):
        self.path = path
        m = json.loads((path / "zarr.json").read_text())
        self.shape = tuple(m["shape"])
        self.chunks = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
        self.codecs = m["codecs"]
        wf = [c for c in self.codecs if c["name"] == "warpfield"]
        self.cfg = Config.from_dict(wf[0]["configuration"]) if wf else None
        self.zstd = zstandard.ZstdDecompressor()

    def raw(self, index) -> bytes | None:
        p = self.path.joinpath("c", *map(str, index))
        return p.read_bytes() if p.exists() else None

    def region(self, index) -> tuple[slice, ...]:
        return tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(index, self.chunks, self.shape, strict=True)
        )

    def read_lossless(self, index) -> np.ndarray:
        sl = self.region(index)
        b = self.raw(index)
        if b is None:
            return np.zeros([s.stop - s.start for s in sl], np.float32)
        a = np.frombuffer(self.zstd.decompress(b), "<f4").reshape(self.chunks)
        return a[tuple(slice(0, s.stop - s.start) for s in sl)].astype(np.float32)


class Acc:
    """Running error, Jacobian and seam statistics for one codec setting."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bytes = 0
        self.enc_s = self.dec_s = 0.0
        self.max = self.ss = self.n = 0.0
        self.viol = 0
        self.det_min = np.inf
        self.seam = np.zeros(2)  # sum |error jump| across chunk boundaries, count
        self.inner = np.zeros(2)  # the same inside chunks
        self.seam_max = 0.0  # largest |error jump| across a chunk boundary
        self.prev_plane = None  # last decoded plane of the previous chunk row

    def add_row(self, orig_x, dec, k_first):
        """`dec` is one decoded chunk row; `orig_x` the original with the previous
        row's last plane prepended (except for the first row). `k_first` is the
        global index of orig_x[0]."""
        pre = self.prev_plane is not None
        own = orig_x[1:] if pre else orig_x
        e = np.abs(dec.astype(np.float64) - own)
        self.max = max(self.max, float(e.max()))
        self.ss += float((e**2).sum())
        self.n += e.size
        del e
        dec_x = dec if self.prev_plane is None else np.concatenate([self.prev_plane[None], dec])
        self.prev_plane = dec[-1].copy()
        # thin overlapping sub-slabs keep the float64 Jacobian terms small
        for s0 in range(0, dec_x.shape[0] - 1, SUB):
            sl = slice(s0, min(s0 + SUB + 1, dec_x.shape[0]))
            o, d = orig_x[sl], dec_x[sl]
            d0 = jacobian.det_jacobian(o, 3, (0, 1, 2), SPACING)
            d1 = jacobian.det_jacobian(d, 3, (0, 1, 2), SPACING)
            self.viol += int(jacobian.violations(d0, d1, MIN_DET).sum())
            self.det_min = min(self.det_min, float(d1.min()))
            ex = d - o  # float32: errors are ~1e-3 mm, far above float32 resolution
            # in-plane jumps: count each plane once (sub-slabs share their end
            # planes, and the prepended plane belongs to the previous row)
            a = 1 if (s0 == 0 and pre) else 0
            b = ex.shape[0] if sl.stop == dec_x.shape[0] else ex.shape[0] - 1
            for ax in range(3):
                jump = np.abs(np.diff(ex if ax == 0 else ex[a:b], axis=ax))
                if ax == 0:
                    lo = k_first + s0 + np.arange(jump.shape[0])
                else:
                    lo = np.arange(jump.shape[ax])
                across = (lo + 1) % self.chunks[ax] == 0
                shp = [1, 1, 1, 1]
                shp[ax] = -1
                m = np.broadcast_to(across.reshape(shp), jump.shape)
                bsum = float(jump[m].sum()) if across.any() else 0.0
                if across.any():
                    self.seam_max = max(self.seam_max, float(jump[m].max()))
                nb = int(across.sum()) * (jump.size // jump.shape[ax])
                self.seam += (bsum, nb)
                self.inner += (float(jump.sum()) - bsum, jump.size - nb)

    def result(self, raw_bytes, total_bytes):
        return dict(
            bytes=self.bytes,
            ratio_vs_raw=raw_bytes / max(self.bytes, 1),
            max_err_um=self.max * 1e3,
            rms_err_um=float(np.sqrt(self.ss / self.n)) * 1e3,
            jacobian_violations=self.viol,
            det_min=self.det_min,
            seam_ratio=float((self.seam[0] / self.seam[1]) / (self.inner[0] / self.inner[1])),
            seam_mean_jump_um=float(self.seam[0] / self.seam[1]) * 1e3,
            inner_mean_jump_um=float(self.inner[0] / self.inner[1]) * 1e3,
            seam_max_jump_um=self.seam_max * 1e3,
            encode_1t_s=self.enc_s,
            decode_1t_s=self.dec_s,
            encode_1t_MBps=total_bytes / self.enc_s / 1e6 if self.enc_s else None,
            decode_1t_MBps=total_bytes / self.dec_s / 1e6,
        )


SUB = 8
SPACING = (0.025, 0.025, 0.025)


def zfp_chunk(blk: np.ndarray, tol: float, acc: Acc) -> np.ndarray:
    out = np.empty_like(blk)
    for c in range(blk.shape[-1]):
        a = np.ascontiguousarray(blk[..., c])
        t0 = time.perf_counter()
        b = zfpy.compress_numpy(a, tolerance=tol)
        t1 = time.perf_counter()
        out[..., c] = zfpy.decompress_numpy(b)
        acc.enc_s += t1 - t0
        acc.dec_s += time.perf_counter() - t1
        acc.bytes += len(b)
    return out


def run_field(out: Path, field: str) -> dict:
    src = Store(out / f"{field}_zstd19-lossless.zarr")
    wf = {e: Store(out / f"{field}_warpfield-eps{e}.zarr") for e in EPS}
    shape, ch = src.shape, src.chunks
    grid = [-(-s // c) for s, c in zip(shape, ch, strict=True)]
    accs = {("warpfield", e): Acc(ch) for e in EPS}
    accs |= {("zfp", t): Acc(ch) for t in ZFP_TOL_MM}
    raw_bytes = int(np.prod(shape)) * 4
    prev_orig = None
    for kc in range(grid[0]):
        k0, k1 = kc * ch[0], min((kc + 1) * ch[0], shape[0])
        rows = [(jc, ic) for jc in range(grid[1]) for ic in range(grid[2])]
        orig = np.zeros((k1 - k0,) + shape[1:], np.float32)
        for jc, ic in rows:
            sl = src.region((kc, jc, ic, 0))
            orig[(slice(None),) + sl[1:]] = src.read_lossless((kc, jc, ic, 0))
        orig_x = orig if prev_orig is None else np.concatenate([prev_orig[None], orig])
        k_first = k0 if prev_orig is None else k0 - 1
        for key, acc in accs.items():
            codec, knob = key
            dec = np.zeros_like(orig)
            for jc, ic in rows:
                index = (kc, jc, ic, 0)
                sl = src.region(index)
                dst = (slice(None),) + sl[1:]
                if codec == "zfp":
                    dec[dst] = zfp_chunk(orig[dst], knob, acc)
                    continue
                s = wf[knob]
                b = s.raw(index)
                if b is None:
                    continue  # all fill
                acc.bytes += len(b)
                t0 = time.perf_counter()
                d = core.decode(b, s.chunks, np.float32, s.cfg, backend="rust")
                acc.dec_s += time.perf_counter() - t0
                dec[dst] = d[tuple(slice(0, x.stop - x.start) for x in sl)]
            acc.add_row(orig_x, dec, k_first)
            del dec
        prev_orig = orig[-1].copy()
        print(f"  {field}: chunk row {kc + 1}/{grid[0]}", flush=True)
    total = int(np.prod(shape)) * 4
    rows = []
    for (codec, knob), acc in accs.items():
        r = dict(
            codec=codec, setting=knob if codec == "warpfield" else f"tol {knob * 1e3:g} um"
        )
        r |= acc.result(raw_bytes, total)
        rows.append(r)
        print(
            f"{field:8s} {codec:9s} {r['setting']:12s} {r['bytes'] / 1e6:8.2f} MB "
            f"{r['ratio_vs_raw']:7.0f}x  max {r['max_err_um']:6.2f} um  rms "
            f"{r['rms_err_um']:5.2f} um  jac {r['jacobian_violations']:6d}  "
            f"seam {r['seam_ratio']:5.2f} (mean {r['seam_mean_jump_um']:.3f}, max "
            f"{r['seam_max_jump_um']:.2f} um)  dec {r['decode_1t_MBps']:5.0f} MB/s"
            + (f"  enc {r['encode_1t_MBps']:5.0f} MB/s" if r["encode_1t_MBps"] else ""),
            flush=True,
        )
    return dict(shape=list(shape), raw_bytes=raw_bytes, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fields", nargs="+", default=["forward", "inverse"])
    a = ap.parse_args()
    res = {f: run_field(a.out, f) for f in a.fields}
    (a.out / "zfp_compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
