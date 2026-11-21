"""The `warpfield` array -> bytes codec for zarr-python 3.

Installing this package registers the codec through the `zarr.codecs` entry
point; `register` is kept for use from a source checkout.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

from warpfield.core import Config, WarpFieldSpecError, decode, encode

CODEC_NAME = "warpfield"


@dataclass(frozen=True)
class WarpFieldCodec(ArrayBytesCodec):
    """Error-bounded, seam-free lossy codec for displacement fields."""

    is_fixed_size = False

    config: Config

    def __init__(
        self,
        *,
        axis_roles: Any,
        error_bound: float = 0.25,
        face_error_bound: float | None = None,
        predictor: str = "eno",
        boundary: str = "clamped",
        jacobian: Any = None,
    ) -> None:
        cfg = Config(
            axis_roles=tuple(axis_roles),
            error_bound=float(error_bound),
            face_error_bound=None if face_error_bound is None else float(face_error_bound),
            predictor=predictor,
            boundary=boundary,
            jacobian=jacobian,
        )
        object.__setattr__(self, "config", cfg)

    @classmethod
    def from_config(cls, cfg: Config) -> WarpFieldCodec:
        return cls(**cfg.to_dict())

    # -- metadata ----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, cfg = parse_named_configuration(data, CODEC_NAME, require_configuration=True)
        return cls(**(cfg or {}))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": CODEC_NAME, "configuration": self.config.to_dict()}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    def validate(self, *, shape, dtype, chunk_grid) -> None:  # type: ignore[override]
        roles = self.config.axis_roles
        if len(shape) != len(roles):
            raise WarpFieldSpecError(
                f"array rank {len(shape)} does not match {len(roles)} axis_roles"
            )
        native = dtype.to_native_dtype() if hasattr(dtype, "to_native_dtype") else dtype
        if np.dtype(native) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise WarpFieldSpecError(f"data type must be float32 or float64, not {native}")
        cs = getattr(chunk_grid, "chunk_shape", None)
        if cs is None:
            return
        if "component" in roles:
            a = roles.index("component")
            if cs[a] != shape[a]:
                raise WarpFieldSpecError(
                    "the chunk must cover the `component` dimension in full "
                    f"(chunk {cs[a]} vs array {shape[a]})"
                )
        j = self.config.jacobian
        if j is not None:
            a = roles.index("component")
            ns = roles.count("spatial")
            if shape[a] != ns:
                raise WarpFieldSpecError(
                    f"jacobian needs one component per spatial axis ({ns}), "
                    f"but the component dimension has extent {shape[a]}"
                )

    # -- codec -------------------------------------------------------------
    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = np.asarray(chunk_array.as_ndarray_like())
        # the Rust backend releases the GIL, so chunks code in parallel threads
        buf = await asyncio.to_thread(encode, arr, self.config)
        return chunk_spec.prototype.buffer.from_bytes(buf)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        out = await asyncio.to_thread(
            decode, chunk_bytes.to_bytes(), tuple(chunk_spec.shape), dtype, self.config
        )
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError("warpfield produces variable-size chunks")


def register() -> None:
    """Register the codec explicitly (not needed when the package is installed)."""
    register_codec(CODEC_NAME, WarpFieldCodec)
