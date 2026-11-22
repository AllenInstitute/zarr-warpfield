"""warpfield vs SPERR, SZ3, HPEZ and QPET-HPEZ on displacement fields.

    uv run python scripts/bench_external.py prepare DATA \\
        [--ants-lossless STORE] [--microns FIELD.npy]
    uv run python scripts/bench_external.py run DATA ants microns synth3d [--only warpfield]

All codecs run single-threaded at the same absolute bound per component.
`results.json` in DATA keeps every row; `--only PREFIX` reruns just the codecs
whose label starts with PREFIX and keeps the others.

Data (DATA/NAME.npy, (Z, Y, X, 3) float32, plus one raw .f32 file per
component for the command-line codecs, and DATA/meta.json):

- ants: the central 256^3 crop of the forward ANTs field
  (examples/ants_mouse_templates, `<field>_zstd19-lossless.zarr` from verify.py).
- microns: the forward MICrONS field as sampled by
  examples/microns_coreg/make_fields.py, before encoding (any .npy works).
- synth3d: `warpfield.synth.preset("3d", "full")`.

The other codecs, built single-threaded (paths can be overridden with the
SPERR3D, HPEZ, HPEZ_VEC and QPET_QOI_CONFIG environment variables):

- SPERR: github.com/NCAR/SPERR, `cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OMP=OFF`;
  binary `sperr3d`.
- SZ3 and HPEZ: the `hpez` binary of github.com/JLiu-1/QPET-Artifact, branch
  `szfamily_qpet_revision` (`-q 0` is SZ3's interpolation, `-q 3` is HPEZ).
- QPET-HPEZ: `hpez_vec` from branch `szfamily_qpet_vec_revision`, with the QoI
  `qoi_configs/sqrt(x^2+y^2+z^2).config` (|u|). Both QPET builds need GMP
  (6.3.0, `--enable-cxx`) and SymEngine; on macOS pass the GMP library
  directory to the linker (`-L`).
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from warpfield import Config, JacobianConfig, jacobian

TOOLS = Path(os.environ.get("BENCH_TOOLS", "."))
SPERR = Path(os.environ.get("SPERR3D", TOOLS / "sperr/build/bin/sperr3d"))
HPEZ = Path(os.environ.get("HPEZ", TOOLS / "install_qpet/bin/hpez"))
HPEZ_VEC = Path(os.environ.get("HPEZ_VEC", TOOLS / "install_qpet_vec/bin/hpez_vec"))
QOI_CFG = Path(
    os.environ.get("QPET_QOI_CONFIG", TOOLS / "qpet_vec/qoi_configs/sqrt(x^2+y^2+z^2).config")
)
MIN_DET = 0.1
TILE = 64
DATA = Path(".")


def run(cmd):
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-500:]} {r.stdout[-500:]}")
    return dt, r.stdout


def seam_ratio(err, tile):
    """mean |error jump| across tile boundaries / mean jump inside tiles."""
    b = i = 0.0
    nb = ni = 0
    for ax in range(3):
        d = np.abs(np.diff(err, axis=ax)).max(axis=-1)
        n = d.shape[ax]
        across = (np.arange(n) + 1) % tile == 0
        sh = [1, 1, 1]
        sh[ax] = -1
        m = np.broadcast_to(across.reshape(sh), d.shape)
        b += d[m].sum()
        nb += m.sum()
        i += d[~m].sum()
        ni += (~m).sum()
    return float((b / max(nb, 1)) / (i / max(ni, 1)))


def metrics(u, dec, eps, spacing, d0, seam_tile):
    err = dec.astype(np.float64) - u
    a = np.abs(err)
    mag = np.abs(np.linalg.norm(dec.astype(np.float64), axis=-1) - np.linalg.norm(u, axis=-1))
    d1 = jacobian.det_jacobian(dec, 3, (0, 1, 2), (spacing,) * 3)
    viol = int(jacobian.violations(d0, d1, MIN_DET).sum())
    return {
        "max_err": float(a.max()),
        "max_err_over_eps": float(a.max() / eps),
        "rms_err": float(np.sqrt((err**2).mean())),
        "qoi_mag_max_err": float(mag.max()),
        "jacobian_violations": viol,
        "seam_ratio": seam_ratio(err, seam_tile) if seam_tile else None,
    }


def bench_warpfield(u, eps, spacing, *, jac=True, whole=False, level=19, tile=None):
    """warpfield, tile by tile (64^3 x 3, or the whole array as one tile)."""
    import json as _json

    import warpfield_rs

    from warpfield.tiles import _padded, iter_tiles, tile_slices

    cfg = Config(
        ("spatial", "spatial", "spatial", "component"),
        eps,
        jacobian=JacobianConfig((spacing,) * 3, MIN_DET) if jac else None,
    )
    cj = _json.dumps(cfg.to_dict())
    t = tile or TILE
    tile = tuple(u.shape) if whole else (t, t, t, 3)
    bufs = {}
    te = 0.0
    for index in iter_tiles(u.shape, tile):
        blk = _padded(np.ascontiguousarray(u[tile_slices(index, u.shape, tile)]), tile, 0.0)
        t0 = time.perf_counter()
        bufs[index] = warpfield_rs.encode(blk, cj, zstd_level=level)[0]
        te += time.perf_counter() - t0
    dec = np.empty_like(u)
    td = 0.0
    for index, buf in bufs.items():
        sl = tile_slices(index, u.shape, tile)
        t0 = time.perf_counter()
        d = warpfield_rs.decode(buf, list(tile), "float32", cj)
        td += time.perf_counter() - t0
        dec[sl] = d[tuple(slice(0, x.stop - x.start) for x in sl)]
    return sum(len(b) for b in bufs.values()), te, td, dec, None if whole else t


def bench_sperr(name, u, eps, chunk, tmp):
    nk, nj, ni = u.shape[:3]
    size = te = td = 0.0
    dec = np.empty_like(u)
    for c in range(3):
        src = DATA / f"{name}_c{c}.f32"
        bs, out = tmp / f"s{c}.sperr", tmp / f"s{c}.out"
        dt, _ = run(
            [
                SPERR,
                "-c",
                "--ftype",
                32,
                "--dims",
                ni,
                nj,
                nk,
                "--chunks",
                chunk,
                chunk,
                chunk,
                "--pwe",
                eps,
                "--bitstream",
                bs,
                src,
            ]
        )
        te += dt
        dt, _ = run([SPERR, "-d", "--decomp_f", out, bs])
        td += dt
        size += bs.stat().st_size
        dec[..., c] = np.fromfile(out, np.float32).reshape(nk, nj, ni)
    return int(size), te, td, dec, chunk


def bench_hpez(name, u, eps, level, tmp):
    nk, nj, ni = u.shape[:3]
    size = te = td = 0.0
    dec = np.empty_like(u)
    for c in range(3):
        src = DATA / f"{name}_c{c}.f32"
        z, out = tmp / f"h{c}.z", tmp / f"h{c}.out"
        dt, _ = run(
            [HPEZ, "-q", level, "-f", "-i", src, "-z", z, "-3", ni, nj, nk, "-M", "ABS", eps]
        )
        te += dt
        dt, _ = run(
            [HPEZ, "-q", level, "-f", "-z", z, "-o", out, "-3", ni, nj, nk, "-M", "ABS", eps]
        )
        td += dt
        size += z.stat().st_size
        dec[..., c] = np.fromfile(out, np.float32).reshape(nk, nj, ni)
    return int(size), te, td, dec, None


def bench_qpet(name, u, eps, tau, tmp):
    """QPET-HPEZ, combined compress + decompress run (its CLI cannot write the
    compressed files separately); ratio and times as it reports them."""
    nk, nj, ni = u.shape[:3]
    src = [DATA / f"{name}_c{c}.f32" for c in range(3)]
    out = [tmp / f"q{c}.out" for c in range(3)]
    _, log = run(
        [
            HPEZ_VEC,
            "-q",
            3,
            "-f",
            "-a",
            "-i",
            *src,
            "-o",
            *out,
            "-3",
            ni,
            nj,
            nk,
            "-M",
            "ABS",
            eps,
            "-m",
            "ABS",
            "-e",
            tau,
            "-c",
            QOI_CFG,
        ]
    )
    ratio = float(re.search(r"Overall compression ratio = ([0-9.eE+-]+)", log).group(1))
    te = float(re.search(r"(?<!de)compression time = ([0-9.eE+-]+)", log).group(1))
    td = float(re.search(r"decompression time = ([0-9.eE+-]+)", log).group(1))
    dec = np.stack([np.fromfile(o, np.float32).reshape(nk, nj, ni) for o in out], -1)
    return int(u.nbytes / ratio), te, td, dec, None


def bench_warpfield_tmp(u, eps, spacing, tmp, **kw):
    return bench_warpfield(u, eps, spacing, **kw)


def prepare(a) -> None:
    import zarr

    from warpfield import synth

    a.data.mkdir(parents=True, exist_ok=True)
    meta_path = a.data / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def save(name, u, spacing, unit, eps):
        u = np.ascontiguousarray(u, dtype=np.float32)
        np.save(a.data / f"{name}.npy", u)
        for c in range(3):
            u[..., c].tofile(a.data / f"{name}_c{c}.f32")
        meta[name] = {"shape": list(u.shape[:3]), "spacing": spacing, "unit": unit, "eps": eps}

    if a.ants_lossless:
        f = zarr.open_array(a.ants_lossless, mode="r")
        o = [(f.shape[i] - 256) // 2 for i in range(3)]
        crop = f[o[0] : o[0] + 256, o[1] : o[1] + 256, o[2] : o[2] + 256]
        save("ants", crop, 0.025, "mm", [0.001, 0.0025, 0.00625])
    if a.microns:
        save("microns", np.load(a.microns), 5.0, "um", [0.1, 0.5, 1.0])
    p = synth.preset("3d", "full")
    save("synth3d", p.full()[0], float(p.spacing), "voxel", [0.05, 0.25])
    meta_path.write_text(json.dumps(meta, indent=1))


def run_bench(a) -> None:
    global DATA
    DATA = a.data
    meta = json.loads((DATA / "meta.json").read_text())
    out = DATA / "results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    only = a.only
    for name in a.datasets:
        m = meta[name]
        u = np.load(DATA / f"{name}.npy")
        raw = u.nbytes
        d0 = jacobian.det_jacobian(u, 3, (0, 1, 2), (m["spacing"],) * 3)
        old = results.get(name, {}).get("runs", [])
        keep = [r for r in old if only and not r["codec"].startswith(only)]
        results[name] = {"shape": m["shape"], "unit": m["unit"], "runs": keep}
        for eps in m["eps"]:
            sp = m["spacing"]
            # (label, function, positional arguments before the scratch directory, keywords)
            codecs = [
                ("warpfield 64^3", bench_warpfield_tmp, (u, eps, sp), {}),
                ("warpfield 64^3 zstd-3", bench_warpfield_tmp, (u, eps, sp), {"level": 3}),
                ("warpfield 64^3 no-jac", bench_warpfield_tmp, (u, eps, sp), {"jac": False}),
                ("warpfield 128^3", bench_warpfield_tmp, (u, eps, sp), {"tile": 128}),
                ("warpfield whole", bench_warpfield_tmp, (u, eps, sp), {"whole": True}),
                ("SPERR 256^3", bench_sperr, (name, u, eps, 256), {}),
                ("SPERR 64^3", bench_sperr, (name, u, eps, 64), {}),
                ("SZ3", bench_hpez, (name, u, eps, 0), {}),
                ("HPEZ", bench_hpez, (name, u, eps, 3), {}),
                ("QPET-HPEZ |u|<=eps", bench_qpet, (name, u, eps, eps), {}),
            ]
            if only:
                codecs = [c for c in codecs if c[0].startswith(only)]
            for label, fn, args, kw in codecs:
                with tempfile.TemporaryDirectory() as t:
                    try:
                        size, te, td, dec, tile = fn(*args, Path(t), **kw)
                    except Exception as e:  # noqa: BLE001
                        print(f"{name} eps={eps} {label}: FAILED {e}", flush=True)
                        continue
                row = {
                    "codec": label,
                    "eps": eps,
                    "bytes": size,
                    "ratio": raw / size,
                    "encode_MBps": raw / te / 1e6,
                    "decode_MBps": raw / td / 1e6,
                }
                row |= metrics(u, dec, eps, m["spacing"], d0, tile)
                results[name]["runs"].append(row)
                print(
                    f"{name:8s} eps={eps:<8g} {label:22s} ratio {row['ratio']:8.1f}  "
                    f"rms/eps {row['rms_err'] / eps:.3f}  jac {row['jacobian_violations']:6d}  "
                    f"enc {row['encode_MBps']:6.1f}  dec {row['decode_MBps']:6.1f} MB/s",
                    flush=True,
                )
                out.write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("data", type=Path)
    p.add_argument("--ants-lossless", type=Path)
    p.add_argument("--microns", type=Path)
    p.set_defaults(fn=prepare)
    r = sub.add_parser("run")
    r.add_argument("data", type=Path)
    r.add_argument("datasets", nargs="+")
    r.add_argument("--only", default=None)
    r.set_defaults(fn=run_bench)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
