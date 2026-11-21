"""Tile-by-tile coding without zarr (warpfield.tiles)."""

import numpy as np
import pytest

from warpfield import Config, JacobianConfig, WarpFieldSpecError, decode, encode
from warpfield.tiles import (
    decode_tiles,
    encode_tiles,
    iter_tiles,
    tile_grid,
    tile_slices,
    zarr_chunk_key,
)

SHAPE = (37, 45, 50, 3)  # no axis divides evenly: every edge is a partial tile
TILES = (16, 16, 16, 3)
CFG = Config(
    axis_roles=("spatial", "spatial", "spatial", "component"),
    error_bound=0.05,
    jacobian=JacobianConfig(grid_spacing=(1.0, 1.0, 1.0), min_det=0.1),
)


def field(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    k, j, i = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape[:3]], indexing="ij")
    comps = []
    for c in range(3):
        a = rng.normal(size=3) * 0.1
        comps.append(
            2.0 * np.sin(a[0] * k + a[1] * j + a[2] * i + c) + 0.01 * rng.normal(size=k.shape)
        )
    return np.stack(comps, -1).astype(np.float32)


class CountingSource:
    """Array-like source that records each read, to check tiles are read one at a time."""

    def __init__(self, arr):
        self.arr, self.shape, self.dtype, self.reads = arr, arr.shape, arr.dtype, []

    def __getitem__(self, sl):
        block = self.arr[sl]
        self.reads.append(block.shape)
        return block


class CountingMapping(dict):
    def __init__(self, *args):
        super().__init__(*args)
        self.gets = []

    def get(self, key, default=None):
        self.gets.append(key)
        return super().get(key, default)


def test_grid_and_slices():
    assert tile_grid(SHAPE, TILES) == (3, 3, 4, 1)
    assert len(list(iter_tiles(SHAPE, TILES))) == 36
    assert tile_slices((2, 2, 3, 0), SHAPE, TILES) == (
        slice(32, 37),
        slice(32, 45),
        slice(48, 50),
        slice(0, 3),
    )
    assert zarr_chunk_key((2, 0, 1, 0)) == "c/2/0/1/0"


def test_roundtrip_within_bound():
    u = field()
    tiles = encode_tiles(u, TILES, CFG)
    assert set(tiles) == set(iter_tiles(SHAPE, TILES))
    out = decode_tiles(tiles, SHAPE, TILES, np.float32, CFG)
    assert out.shape == SHAPE
    assert np.abs(out - u).max() <= CFG.error_bound


def test_tiles_are_the_padded_chunks():
    u = field()
    tiles = encode_tiles(u, TILES, CFG, fill_value=0.0)
    for index, buf in tiles.items():
        sl = tile_slices(index, SHAPE, TILES)
        block = np.zeros(TILES, np.float32)
        block[tuple(slice(0, s.stop - s.start) for s in sl)] = u[sl]
        assert buf == encode(block, CFG)
        assert np.array_equal(
            decode(buf, TILES, np.float32, CFG),
            decode(encode(block, CFG), TILES, np.float32, CFG),
        )


def test_reads_one_tile_at_a_time():
    src = CountingSource(field())
    encode_tiles(src, TILES, CFG)
    assert len(src.reads) == 36
    assert max(int(np.prod(r)) for r in src.reads) <= int(np.prod(TILES))


def test_region_reads_only_overlapping_tiles():
    u = field()
    tiles = CountingMapping(encode_tiles(u, TILES, CFG))
    full = decode_tiles(tiles, SHAPE, TILES, np.float32, CFG)
    tiles.gets.clear()
    region = (slice(10, 20), slice(30, 45), slice(0, 5), slice(None))
    part = decode_tiles(tiles, SHAPE, TILES, np.float32, CFG, region=region)
    assert np.array_equal(part, full[region])
    # k: tiles 0-1, j: tiles 1-2, i: tile 0, c: tile 0
    assert sorted(tiles.gets) == sorted((k, j, 0, 0) for k in (0, 1) for j in (1, 2))


def test_missing_tiles_and_skip_fill():
    u = field()
    u[:16, :16, :16] = 0.0  # tile (0, 0, 0, 0) is all fill
    tiles = encode_tiles(u, TILES, CFG, skip_fill=True)
    assert (0, 0, 0, 0) not in tiles
    out = decode_tiles(tiles, SHAPE, TILES, np.float32, CFG)
    assert np.all(out[:16, :16, :16] == 0.0)
    assert np.abs(out - u).max() <= CFG.error_bound


def test_parallel_matches_serial():
    u = field()
    serial = encode_tiles(u, TILES, CFG)
    parallel = encode_tiles(u, TILES, CFG, workers=4)
    assert serial == parallel
    a = decode_tiles(serial, SHAPE, TILES, np.float32, CFG)
    b = decode_tiles(serial, SHAPE, TILES, np.float32, CFG, workers=4)
    assert np.array_equal(a, b)


def test_component_axis_must_be_whole():
    with pytest.raises(WarpFieldSpecError, match="component"):
        encode_tiles(field(), (16, 16, 16, 2), CFG)


def test_bytes_match_zarr_chunks(tmp_path):
    zarr = pytest.importorskip("zarr")
    from zarr_warpfield import WarpFieldCodec

    u = field()
    path = tmp_path / "field.zarr"
    z = zarr.create_array(
        str(path),
        shape=SHAPE,
        chunks=TILES,
        dtype="float32",
        fill_value=0.0,
        serializer=WarpFieldCodec.from_config(CFG),
        compressors=None,
    )
    z[...] = u
    tiles = encode_tiles(u, TILES, CFG, key=zarr_chunk_key)
    on_disk = {
        str(p.relative_to(path)): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    }
    assert on_disk == tiles
    # and tiles written under zarr's keys read back through zarr
    assert np.array_equal(
        z[...],
        decode_tiles(on_disk, SHAPE, TILES, np.float32, CFG, key=zarr_chunk_key),
    )
