import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from zarr_warpfield import WarpFieldCodec, WarpFieldSpecError, synth  # noqa: E402
from zarr_warpfield.codec import CODEC_NAME  # noqa: E402


def make(store, u, chunks, codec, shards=None):
    a = zarr.create_array(
        store,
        shape=u.shape,
        chunks=chunks,
        shards=shards,
        dtype=u.dtype,
        serializer=codec,
        compressors=None,
        fill_value=0.0,
    )
    a[...] = u
    return zarr.open_array(store)


@pytest.mark.parametrize("name", synth.PRESET_NAMES)
def test_zarr_roundtrip(tmp_path, name):
    p = synth.preset(name, "small")
    u, _ = p.full()
    codec = WarpFieldCodec(axis_roles=p.axis_roles, error_bound=0.25)
    b = make(zarr.storage.LocalStore(tmp_path / "a.zarr"), u, p.chunks, codec)
    assert np.abs(b[...] - u).max() <= 0.25
    # a partial read crossing chunk boundaries
    sl = tuple(slice(1, n - 1) for n in u.shape[:-1])
    assert np.abs(b[sl] - u[sl]).max() <= 0.25


def test_sharded(tmp_path):
    p = synth.preset("3d", "small")
    u, _ = p.full()
    codec = WarpFieldCodec(
        axis_roles=p.axis_roles,
        error_bound=0.1,
        jacobian={"grid_spacing": [4, 4, 4], "min_det": 0.1},
    )
    b = make(
        zarr.storage.LocalStore(tmp_path / "s.zarr"), u, p.chunks, codec, shards=(48, 48, 48, 3)
    )
    assert np.abs(b[...] - u).max() <= 0.1


def test_metadata_roundtrip():
    c = WarpFieldCodec(
        axis_roles=["smooth", "spatial", "spatial", "component"],
        error_bound=0.2,
        predictor="cubic",
        jacobian={"grid_spacing": [16, 16], "min_det": 0.0},
    )
    d = c.to_dict()
    assert d["name"] == CODEC_NAME
    c2 = WarpFieldCodec.from_dict(d)
    assert c2.to_dict() == d


def test_rejects_split_component(tmp_path):
    with pytest.raises(WarpFieldSpecError):
        zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=(32, 32, 2),
            chunks=(16, 16, 1),
            dtype="float32",
            serializer=WarpFieldCodec(axis_roles=["spatial", "spatial", "component"]),
            compressors=None,
        )


def test_rejects_component_mismatch_for_jacobian():
    with pytest.raises(WarpFieldSpecError):
        zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=(32, 32, 3),
            chunks=(16, 16, 3),
            dtype="float32",
            serializer=WarpFieldCodec(
                axis_roles=["spatial", "spatial", "component"],
                jacobian={"grid_spacing": [1, 1]},
            ),
            compressors=None,
        )


def test_rejects_int_dtype():
    with pytest.raises(WarpFieldSpecError):
        zarr.create_array(
            zarr.storage.MemoryStore(),
            shape=(32, 32, 2),
            chunks=(16, 16, 2),
            dtype="int16",
            serializer=WarpFieldCodec(axis_roles=["spatial", "spatial", "component"]),
            compressors=None,
        )
