"""The `warpfield` codec for zarr v3 (zarr-python 3).

Installing this package registers the codec through the `zarr.codecs` entry
point. The format itself lives in the `warpfield` package, which does not need
zarr; its API is re-exported here for code written against earlier versions.
"""

import importlib

from warpfield import *  # noqa: F403
from warpfield import __all__ as _core_all
from zarr_warpfield.codec import CODEC_NAME, WarpFieldCodec, register

__all__ = [*_core_all, "CODEC_NAME", "WarpFieldCodec", "register"]


def __getattr__(name):
    # `zarr_warpfield.synth`, `.jacobian`, ...: the core modules.
    try:
        return importlib.import_module(f"warpfield.{name}")
    except ModuleNotFoundError:
        raise AttributeError(name) from None
