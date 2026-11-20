"""Choose between the Rust extension (`zarr_warpfield_rs`) and the numpy reference.

Both produce identical code streams and bit-identical decoded values. The Rust
backend is used automatically when the extension is installed
(`pip install zarr-warpfield-rs`, or `zarr-warpfield[fast]`); set
`ZARR_WARPFIELD_BACKEND=python` to force the reference, or pass `backend=` to
`encode` / `decode`.
"""

from __future__ import annotations

import os

try:  # pragma: no cover - depends on the environment
    import zarr_warpfield_rs as rs
except ImportError:  # pragma: no cover
    rs = None

BACKENDS = ("auto", "rust", "python")


def available() -> bool:
    return rs is not None


def resolve(backend: str | None) -> str:
    b = backend or os.environ.get("ZARR_WARPFIELD_BACKEND", "auto")
    if b not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {b!r}")
    if b == "auto":
        return "rust" if rs is not None else "python"
    if b == "rust" and rs is None:
        raise ImportError("the Rust backend needs the zarr-warpfield-rs package")
    return b
