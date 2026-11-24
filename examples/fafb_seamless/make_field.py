"""A cutout of the SEAMLeSS FAFB v14 -> v14.1 alignment field as a warpfield
OME-Zarr store, for viewing v14 images warped into v14.1 in Neuroglancer.

The source field (Seung lab archive, CloudVolume precomputed) holds, per
section, a 2-d displacement field: int16 residuals in units of 1/4 of a
4 nm pixel, i.e. in nanometers, at 16 x 16 nm per sample, one section per chunk;
channel 0 is the y displacement, channel 1 x (see to_components).
SEAMLeSS renders v14.1 by pulling from v14: out(p) = v14(p + u(p)), p and u in
the same absolute 4 nm pixel frame, with fields and images sampled at pixel
centers (torch grid_sample on a size-agnostic identity grid; fields upsampled
with nn.Upsample(bilinear)). That is Neuroglancer's warp, W(x) = M(x + u(x)).

The fields are computed section by section and jump between sections (folds,
cracks, per-section corrections), so the store treats z as `independent`: each
section is coded on its own, with no prediction (and no seam constraint) across
sections, and chunks are one section thick. Shards are one section by
--tile samples (1024: one source chunk), and the work goes tile by tile, so
memory does not grow with the field of view.

Layout (out/<name>.zarr):
  zarr.json   OME-Zarr 0.6 multiscale, axes (c, z, y, x) in nm; c is the RFC-5
              "displacement" axis: (dz, dy, dx), with dz = 0.
  s0/         float32 (3, Z, Y, X) at 40 x 16 x 16 nm; chunks (3, 1, 256, 256),
              shards (3, 1, 1024, 1024); warpfield codec, roles
              (component, independent, spatial, spatial).
  s1.. s3/    2x2 means in-plane only (sections are never mixed): 32, 64, 128 nm
              (--levels 1: none, s0 only); error bound x growth**L
              (--level-error-growth 2: doubled per level, as the spacing).
  attributes.warpfield: error bound, per-level interpolation error (nm), source.

    uv run python examples/fafb_seamless/make_field.py [--center 74067 42119 4384]
        [--sections 100] [--size 2048] [--error-bound 0.5]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

from zarr_warpfield import WarpFieldCodec

FIELD_URL = "https://td.princeton.edu/sseung-archive/fafbv14-vf/v0"
SCALE_KEY = "16_16_40"
FIELD_RES = (16.0, 16.0, 40.0)  # nm, x y z
FIELD_CHUNK = 1024  # x, y; one section per chunk
FIELD_OFFSET = (4096, 0, 1)  # voxel_offset of the 16 nm scale, x y z
CHUNK = 256
N_LEVELS = 4
FETCH_THREADS = 4  # source chunks fetched at once per section (x --threads sections)
OUT = Path(__file__).resolve().parents[2] / "out" / "fafb_seamless"


def fetch(url: str) -> tuple[bytes | None, int]:
    """GET (via curl, which uses the system's certificate store), gunzipped;
    also the bytes transferred. (None, 0) if the chunk does not exist."""
    for attempt in range(5):
        with tempfile.NamedTemporaryFile() as tmp:
            r = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--compressed",
                    "-m",
                    "120",
                    "-o",
                    tmp.name,
                    "-w",
                    "%{http_code} %{size_download}",
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                code, n = r.stdout.split()
                if code == "404":
                    return None, 0
                if code == "200":
                    return Path(tmp.name).read_bytes(), int(n)
        if attempt == 4:
            raise RuntimeError(f"{url}: {r.stdout} {r.stderr}")
        time.sleep(2**attempt)
    raise AssertionError


def read_section(z: int, x0: int, y0: int, n: int) -> tuple[np.ndarray, int, int]:
    """int16 (2, n, n) field (channels; [y, x]) of section z, the region
    [x0, x0 + n) x [y0, y0 + n) of 16 nm samples; also bytes fetched and the
    number of missing source chunks (left 0)."""
    out = np.zeros((2, n, n), np.int16)
    nbytes = missing = 0
    cx = range(
        (x0 - FIELD_OFFSET[0]) // FIELD_CHUNK, -(-(x0 + n - FIELD_OFFSET[0]) // FIELD_CHUNK)
    )
    cy = range(
        (y0 - FIELD_OFFSET[1]) // FIELD_CHUNK, -(-(y0 + n - FIELD_OFFSET[1]) // FIELD_CHUNK)
    )
    corners = [
        (FIELD_OFFSET[0] + i * FIELD_CHUNK, FIELD_OFFSET[1] + j * FIELD_CHUNK)
        for i in cx
        for j in cy
    ]

    def get(corner):
        ax, ay = corner
        key = f"{ax}-{ax + FIELD_CHUNK}_{ay}-{ay + FIELD_CHUNK}_{z}-{z + 1}"
        return corner, *fetch(f"{FIELD_URL}/{SCALE_KEY}/{key}")

    with ThreadPoolExecutor(FETCH_THREADS) as pool:
        for (ax, ay), body, nb in pool.map(get, corners):
            nbytes += nb
            if body is None:
                missing += 1
                continue
            # raw precomputed: Fortran order (x, y, z, c)
            a = np.frombuffer(body, "<i2").reshape(2, 1, FIELD_CHUNK, FIELD_CHUNK)[:, 0]
            lx, hx = max(ax, x0), min(ax + FIELD_CHUNK, x0 + n)
            ly, hy = max(ay, y0), min(ay + FIELD_CHUNK, y0 + n)
            out[:, ly - y0 : hy - y0, lx - x0 : hx - x0] = a[
                :, ly - ay : hy - ay, lx - ax : hx - ax
            ]
    return out, nbytes, missing


def to_components(v: np.ndarray) -> np.ndarray:
    """int16 channels (0, 1) nm -> float32 (dz, dy, dx) nm, shape (3, 1, n, n).

    Channel 0 is the y displacement and channel 1 the x displacement (as stored
    in this volume; SEAMLeSS's docstrings say (x, y)). Determined against the
    published v14.1 render with validate.py: with channel 0 = y, warped v14
    matches v14.1 at NCC 0.998 with no residual shift; with channel 0 = x,
    NCC ~0 and shifts of ~50 px.
    """
    u = np.zeros((3, 1, *v.shape[1:]), np.float32)
    u[1, 0] = v[0]
    u[2, 0] = v[1]
    return u


def det_j(u: np.ndarray, h: float) -> np.ndarray:
    """In-plane det(I + du/dx) of a (3, n, n) section (dz, dy, dx), spacing h nm."""
    dy_y, dy_x = np.gradient(u[1], h)
    dx_y, dx_x = np.gradient(u[2], h)
    return (1 + dx_x) * (1 + dy_y) - dx_y * dy_x


def codec(error_bound: float) -> WarpFieldCodec:
    return WarpFieldCodec(
        axis_roles=["component", "independent", "spatial", "spatial"],
        error_bound=error_bound,
        face_error_bound=error_bound / 4,
    )


def shard_edge(tile: int, extent: int) -> int:
    """The tile, or for a smaller level its extent rounded up to whole chunks."""
    c = min(CHUNK, extent)
    return min(tile, -(-extent // c) * c)


def create(root, path, shape, shard_sections, shard_tile, error_bound):
    """Chunks (3, 1, 256, 256); shards (3, shard_sections, tile, tile), so a
    shard is written from one tile-sized work unit and nothing holds a whole
    section."""
    return root.create_array(
        path,
        shape=shape,
        chunks=(3, 1, min(CHUNK, shape[2]), min(CHUNK, shape[3])),
        shards=(3, shard_sections, *(shard_edge(shard_tile, m) for m in shape[2:])),
        dtype="float32",
        fill_value=0.0,
        serializer=codec(error_bound),
        compressors=None,
        dimension_names=["c", "z", "y", "x"],
    )


def ome(name, levels, z0, y0, x0):
    axes = [
        {"name": "c", "type": "displacement", "discrete": True},
        *({"name": a, "type": "space", "unit": "nanometer"} for a in ("z", "y", "x")),
    ]
    rz, ry, rx = FIELD_RES[2], FIELD_RES[1], FIELD_RES[0]
    datasets = []
    for path, f in levels:
        # level sample n is the mean of level-0 samples f n .. f n + f - 1
        datasets.append(
            {
                "path": path,
                "coordinateTransformations": [
                    {
                        "type": "sequence",
                        "input": {"path": path},
                        "output": {"name": "FAFB v14.1"},
                        "transformations": [
                            {"type": "scale", "scale": [1.0, rz, ry * f, rx * f]},
                            {
                                "type": "translation",
                                # sample centers, as CloudVolume / SEAMLeSS
                                "translation": [
                                    0.0,
                                    rz * (z0 + 0.5),
                                    ry * (y0 + f / 2),
                                    rx * (x0 + f / 2),
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    return {
        "version": "0.6",
        "multiscales": [
            {
                "name": name,
                "coordinateSystems": [{"name": "FAFB v14.1", "axes": axes}],
                "datasets": datasets,
            }
        ],
    }


def level_errors(root, n_levels, windows):
    """Max / p99 deviation (nm) of each level, bilinearly interpolated within
    the section, from s0, over the given (section, y, x, size) windows of s0
    (read window by window, never a whole section)."""
    s0 = root["s0"]
    out = [{"max": 0.0, "p99": 0.0}]
    for level in range(1, n_levels):
        f = 2**level
        a = root[f"s{level}"]
        errs = []
        for k, y, x, t in windows:
            ref = s0[:, k, y : y + t, x : x + t]
            # level-f coordinates of the s0 samples, and the level window
            # (one sample of margin) that interpolating them needs
            cy = (np.arange(y, y + t) - (f - 1) / 2) / f
            cx = (np.arange(x, x + t) - (f - 1) / 2) / f
            ly = max(0, int(np.floor(cy[0])) - 1), min(a.shape[2], int(np.ceil(cy[-1])) + 2)
            lx = max(0, int(np.floor(cx[0])) - 1), min(a.shape[3], int(np.ceil(cx[-1])) + 2)
            lv = a[:, k, ly[0] : ly[1], lx[0] : lx[1]]
            yy, xx = np.meshgrid(cy - ly[0], cx - lx[0], indexing="ij")
            v = np.stack(
                [
                    ndimage.map_coordinates(lv[c], [yy, xx], order=1, mode="nearest")
                    for c in (1, 2)
                ]
            )
            errs.append(np.abs(v - ref[1:]).max(0).ravel())
        e = np.concatenate(errs)
        out.append({"max": float(e.max()), "p99": float(np.percentile(e, 99))})
    return out


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=[74067, 42119, 4384],
        help="x y (4 nm pixels), z (section)",
    )
    ap.add_argument("--sections", type=int, default=100)
    ap.add_argument("--size", type=int, default=2048, help="16 nm samples per side")
    ap.add_argument("--error-bound", type=float, default=0.5, help="nm")
    ap.add_argument("--name", default="fafb_v14_to_v14.1")
    ap.add_argument("--threads", type=int, default=16, help="work units at once")
    ap.add_argument(
        "--tile",
        type=int,
        default=FIELD_CHUNK,
        help="16 nm samples per side of a work unit and of a shard",
    )
    ap.add_argument("--z-block", type=int, default=25, help="sections per work unit")
    ap.add_argument(
        "--levels",
        type=int,
        default=N_LEVELS,
        help="multiscale levels (1: the 16 nm field only)",
    )
    ap.add_argument(
        "--level-error-growth",
        type=float,
        default=1.0,
        help="error bound of level L = error bound * growth**L (2: doubles per level, "
        "as the sample spacing does)",
    )
    a = ap.parse_args()
    t0 = time.time()

    n = a.size
    # region on the field's chunk grid (fewest downloads), around the center
    cx, cy = a.center[0] / 4, a.center[1] / 4
    x0 = int(
        FIELD_OFFSET[0] + round((cx - n / 2 - FIELD_OFFSET[0]) / FIELD_CHUNK) * FIELD_CHUNK
    )
    y0 = int(
        FIELD_OFFSET[1] + round((cy - n / 2 - FIELD_OFFSET[1]) / FIELD_CHUNK) * FIELD_CHUNK
    )
    z0 = int(a.center[2]) - a.sections // 2
    Z = a.sections
    print(f"region (16 nm): x [{x0}, {x0 + n}) y [{y0}, {y0 + n}) z [{z0}, {z0 + Z})")

    OUT.mkdir(parents=True, exist_ok=True)
    store = OUT / f"{a.name}.zarr"
    root = zarr.open_group(store, mode="w", zarr_format=3)
    T = a.tile
    if n % T or T % CHUNK or T % FIELD_CHUNK:
        raise SystemExit(f"--size must be a multiple of --tile, a multiple of {FIELD_CHUNK}")
    s0 = create(root, "s0", (3, Z, n, n), 1, T, a.error_bound)

    stats = {
        "bytes_fetched": 0,
        "missing_chunks": 0,
        "max_err": 0.0,
        "max_abs_u": 0.0,
        "det_min_source": np.inf,
        "det_min_decoded": np.inf,
        "max_dz_jump": 0.0,
    }

    def work(unit):
        """One tile (one source chunk) through a block of sections; the tile
        of the section before the block is read too, for the z jump."""
        ty, tx, k0, k1 = unit
        r = {"nb": 0, "miss": 0, "err": 0.0, "u": 0.0, "dsrc": np.inf, "ddec": np.inf}
        r["jump"] = 0.0
        prev = None
        if k0 > 0:
            prev, nb, _ = read_section(z0 + k0 - 1, x0 + tx, y0 + ty, T)
            r["nb"] += nb
        for k in range(k0, k1):
            v, nb, miss = read_section(z0 + k, x0 + tx, y0 + ty, T)
            u = to_components(v)
            s0[:, k : k + 1, ty : ty + T, tx : tx + T] = u  # one whole shard
            back = s0[:, k, ty : ty + T, tx : tx + T]
            r["nb"] += nb
            r["miss"] += miss
            r["err"] = max(r["err"], float(np.abs(back - u[:, 0]).max()))
            r["u"] = max(r["u"], float(np.abs(v).max()))
            r["dsrc"] = min(r["dsrc"], float(det_j(u[:, 0], 16).min()))
            r["ddec"] = min(r["ddec"], float(det_j(back, 16).min()))
            if prev is not None:
                jump = np.abs(v.astype(np.int32) - prev).max()
                r["jump"] = max(r["jump"], float(jump))
            prev = v
        return unit, r

    units = [
        (ty, tx, k0, min(k0 + a.z_block, Z))
        for k0 in range(0, Z, a.z_block)
        for ty in range(0, n, T)
        for tx in range(0, n, T)
    ]
    with ThreadPoolExecutor(a.threads) as pool:
        for i, ((ty, tx, k0, k1), r) in enumerate(pool.map(work, units)):
            stats["bytes_fetched"] += r["nb"]
            stats["missing_chunks"] += r["miss"]
            stats["max_err"] = max(stats["max_err"], r["err"])
            stats["max_abs_u"] = max(stats["max_abs_u"], r["u"])
            stats["det_min_source"] = min(stats["det_min_source"], r["dsrc"])
            stats["det_min_decoded"] = min(stats["det_min_decoded"], r["ddec"])
            # how much the field jumps between neighbouring sections
            stats["max_dz_jump"] = max(stats["max_dz_jump"], r["jump"])
            print(
                f"  [{i + 1}/{len(units)} {time.time() - t0:.0f} s] z [{z0 + k0}, {z0 + k1}) "
                f"tile y {y0 + ty} x {x0 + tx}: |u| <= {r['u']:.0f} nm, err {r['err']:.3f}, "
                f"det J min {r['ddec']:.3f}"
                + (f", {r['miss']} chunks missing" if r["miss"] else ""),
                flush=True,
            )

    # in-plane 2x2 means of the decoded level below; sections never mixed;
    # tile by tile, as s0
    levels = [("s0", 1)]
    prev_arr, shape = s0, (n, n)
    for level in range(1, a.levels):
        shape = (shape[0] // 2, shape[1] // 2)
        tl = shard_edge(T, min(shape))
        per_section = (-(-tl // CHUNK)) ** 2
        ks = max(1, min(Z, 16 // per_section))
        arr = create(
            root,
            f"s{level}",
            (3, Z, *shape),
            ks,
            tl,
            a.error_bound * a.level_error_growth**level,
        )

        def down(unit, arr=arr, src=prev_arr, shape=shape, tl=tl):
            k0, k1, ty, tx = unit
            h, w = min(tl, shape[0] - ty), min(tl, shape[1] - tx)
            u = src[:, k0:k1, 2 * ty : 2 * (ty + h), 2 * tx : 2 * (tx + w)]
            arr[:, k0:k1, ty : ty + h, tx : tx + w] = u.reshape(3, k1 - k0, h, 2, w, 2).mean(
                axis=(3, 5), dtype=np.float64
            )

        units = [
            (k0, min(k0 + ks, Z), ty, tx)
            for k0 in range(0, Z, ks)
            for ty in range(0, shape[0], tl)
            for tx in range(0, shape[1], tl)
        ]
        with ThreadPoolExecutor(a.threads) as pool:
            list(pool.map(down, units))
        print(f"  s{level} {shape} [{time.time() - t0:.0f} s]", flush=True)
        levels.append((f"s{level}", 2**level))
        prev_arr = arr

    rng = np.random.default_rng(0)
    windows = [
        (int(k), int(rng.integers(n // T)) * T, int(rng.integers(n // T)) * T, T)
        for k in sorted(rng.choice(Z, min(Z, 8), replace=False))
        for _ in range(4)
    ]
    errs = level_errors(root, a.levels, windows)
    root.attrs["ome"] = ome(a.name, levels, z0, y0, x0)
    root.attrs["warpfield"] = {
        "version": "0.2",
        "error_bound": a.error_bound,
        "levels": [
            {
                "path": p,
                "error_bound": a.error_bound * a.level_error_growth**i,
                "interpolation_error": e,
            }
            for i, ((p, _), e) in enumerate(zip(levels, errs, strict=True))
        ],
        "extrapolation": "nearest",
        "source": {
            "url": f"precomputed://{FIELD_URL}",
            "scale": SCALE_KEY,
            "region_16nm": {"x": [x0, x0 + n], "y": [y0, y0 + n], "z": [z0, z0 + Z]},
            "convention": "v14.1(p) = v14(p + u(p)); int16 = nm; channel 0 = y, 1 = x",
        },
    }
    raw = Z * n * n * 2 * 2
    report = {
        "region_16nm": {"x": [x0, x0 + n], "y": [y0, y0 + n], "z": [z0, z0 + Z]},
        "error_bound_nm": a.error_bound,
        "max_decode_error_nm": stats["max_err"],
        "max_abs_displacement_nm": stats["max_abs_u"],
        "max_jump_between_sections_nm": stats["max_dz_jump"],
        "det_j_min_source": stats["det_min_source"],
        "det_j_min_decoded": stats["det_min_decoded"],
        "missing_source_chunks": stats["missing_chunks"],
        "levels": dict(zip([p for p, _ in levels], errs, strict=True)),
        "bytes": {
            "source_int16_raw": raw,
            "source_gzip_fetched": stats["bytes_fetched"],
            "store_total": size(store),
            **{p: size(store / p) for p, _ in levels},
        },
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / f"{a.name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
