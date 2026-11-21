"""The JNI bindings (java/) must be bit-identical to the numpy reference.

Needs a JDK (`java` on PATH, in $JAVA_HOME, or $JAVA), the native library built
by `cargo build --release -p warpfield-jni` in rust/, and the compiled Java
classes (`mvn test-compile` in java/; java/build_native.sh does both).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_wasm import _cases
from warpfield import core
from warpfield import format as fmt

ROOT = Path(__file__).resolve().parents[1]
JAVA_DIR = ROOT / "java"
CLASSES = [JAVA_DIR / "target" / "classes", JAVA_DIR / "target" / "test-classes"]
LIB_DIR = ROOT / "rust" / "target" / "release"


def _java():
    for cand in (
        os.environ.get("JAVA"),
        os.environ.get("JAVA_HOME") and str(Path(os.environ["JAVA_HOME"]) / "bin" / "java"),
        shutil.which("java"),
    ):
        if cand and subprocess.run([cand, "-version"], capture_output=True).returncode == 0:
            return cand
    return None


JAVA = _java()
LIB = next(LIB_DIR.glob("*warpfield_jni.*"), None) if LIB_DIR.exists() else None

pytestmark = pytest.mark.skipif(
    not (JAVA and LIB and all(c.exists() for c in CLASSES)),
    reason="JDK, native library or compiled Java classes not available",
)


def test_java_matches_python(tmp_path):
    cases = list(_cases())
    for name, arr, cfg in cases:
        shape = " ".join(str(s) for s in arr.shape)
        (tmp_path / f"{name}.meta").write_text(
            f"{arr.dtype.name}\n{shape}\n{json.dumps(cfg.to_dict())}\n"
        )
        le = arr.astype(arr.dtype.newbyteorder("<"))
        (tmp_path / f"{name}.raw").write_bytes(le.tobytes())
        (tmp_path / f"{name}.encoded").write_bytes(core.encode(arr, cfg, backend="python"))
    r = subprocess.run(
        [
            JAVA,
            f"-Dwarpfield.library={LIB}",
            "-cp",
            os.pathsep.join(str(c) for c in CLASSES),
            "org.warpfield.CrossCheck",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert f"crosschecked {len(cases)} cases" in r.stdout
    for name, arr, cfg in cases:
        le = arr.dtype.newbyteorder("<")
        uint = np.dtype("<u4") if arr.dtype == np.float32 else np.dtype("<u8")
        py_buf = (tmp_path / f"{name}.encoded").read_bytes()
        ref = core.decode(py_buf, arr.shape, arr.dtype, cfg, backend="python")
        got = np.frombuffer((tmp_path / f"{name}.java_decoded").read_bytes(), le)
        np.testing.assert_array_equal(
            got.view(uint), ref.ravel().astype(le).view(uint), err_msg=name
        )
        # Java-encoded bytes carry the same streams as Python's (zstd framing may differ)
        j_buf = (tmp_path / f"{name}.java_encoded").read_bytes()
        a, b = fmt.read_chunk(py_buf, arr.dtype), fmt.read_chunk(j_buf, arr.dtype)
        assert a[0] == b[0], name
        for x, y in zip(a[1:], b[1:], strict=True):
            np.testing.assert_array_equal(
                np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)
            )
        out = core.decode(j_buf, arr.shape, arr.dtype, cfg, backend="python")
        np.testing.assert_array_equal(out.view(uint), ref.view(uint), err_msg=name)
