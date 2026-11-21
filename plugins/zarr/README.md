# zarr-warpfield

The `warpfield` codec for zarr v3 (zarr-python 3): error-bounded, seam-free
lossy compression of registration displacement fields, as an array-to-bytes
codec. Installing this package registers the codec with zarr.

```python
import zarr
from zarr_warpfield import WarpFieldCodec

codec = WarpFieldCodec(
    axis_roles=["spatial", "spatial", "spatial", "component"],
    error_bound=0.0025,
    jacobian={"grid_spacing": [0.025] * 3, "min_det": 0.1},
)
z = zarr.create_array(
    "field.zarr",
    shape=(592, 384, 520, 3),
    chunks=(64, 64, 64, 3),
    dtype="float32",
    serializer=codec,
    compressors=None,
)
```

The format and the zarr-independent API (`encode`, `decode`, `encode_tiles`,
`decode_tiles`) are in the [`warpfield`](../../README.md) package, which this
one depends on and re-exports.
