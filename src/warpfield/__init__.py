"""Error-bounded, seam-free lossy compression of registration displacement fields.

The format is the `warpfield` codec: one array (a "chunk") per encoded buffer,
with a per-sample error bound, a tighter bound on the chunk faces (so errors do
not jump across chunk boundaries) and, optionally, preserved Jacobian
determinants. Nothing here depends on zarr:

- `encode` / `decode` code one array;
- `encode_tiles` / `decode_tiles` code a large array tile by tile, into and out
  of any mapping (a dict, a directory, a database, ...);
- `zarr-warpfield` registers the same format as a zarr v3 codec.

The Rust implementation (`warpfield-rs`) is used automatically when installed;
it produces identical streams and bit-identical decoded values.
"""

from warpfield.core import (
    Config,
    JacobianConfig,
    WarpFieldSpecError,
    decode,
    encode,
)
from warpfield.format import WarpFieldFormatError
from warpfield.tiles import TileIndex, decode_tiles, encode_tiles, tile_grid

__all__ = [
    "Config",
    "JacobianConfig",
    "TileIndex",
    "WarpFieldFormatError",
    "WarpFieldSpecError",
    "decode",
    "decode_tiles",
    "encode",
    "encode_tiles",
    "tile_grid",
]
