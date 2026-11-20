"""Error-bounded, seam-free lossy compression of registration displacement fields."""

from zarr_warpfield.core import Config, JacobianConfig, WarpFieldSpecError, decode, encode

__all__ = [
    "Config",
    "JacobianConfig",
    "WarpFieldCodec",
    "WarpFieldSpecError",
    "decode",
    "encode",
    "register",
]


def __getattr__(name):
    # zarr is an optional dependency: import the codec lazily.
    if name in ("WarpFieldCodec", "register"):
        from zarr_warpfield import codec

        return getattr(codec, name)
    raise AttributeError(name)
