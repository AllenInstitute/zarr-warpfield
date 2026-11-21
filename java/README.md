# warpfield for Java

JNI bindings to the Rust warpfield codec: error-bounded, seam-free compression
of registration displacement fields, one chunk at a time. Decoded values are
bit-identical to the Python, Rust and WebAssembly decoders (`tests/test_java.py`
checks this).

- **Java 8 or newer.** The classes are Java 8 bytecode, like n5, n5-zarr and
  zarr-java, and use plain JNI, so they need no flags on any JVM.
- **One jar for every platform.** The native library for macOS (arm64,
  x86_64), Linux (x86_64, aarch64; glibc ≥ 2.17) and Windows (x86_64) is packaged
  in the jar and extracted on first use.
- **No dependencies.**

## Use

```java
import org.warpfield.Warpfield;

// the codec's "configuration" object from the zarr metadata, as JSON
String config = "{\"axis_roles\": [\"component\", \"spatial\", \"spatial\", \"spatial\"],"
    + " \"error_bound\": 0.25}";
int[] shape = {3, 64, 64, 64};                       // one chunk, C order

byte[] encoded = Warpfield.encode(field, shape, config);       // float[] or double[]
float[] u = Warpfield.decodeFloat32(encoded, shape, config);

// into an existing array (no allocation), from part of a byte[]
Warpfield.decode(buf, offset, length, shape, config, out, outOffset);

// between direct buffers (native byte order), e.g. memory-mapped files
Warpfield.decode(srcByteBuffer, shape, config, dstFloatBuffer);
```

Errors (bad configuration, wrong shape or sizes, corrupt data) throw
`org.warpfield.WarpfieldException`; nothing reads or writes out of bounds.

**Axis order.** Arrays are flat in C order (last axis fastest), the order of
`axis_roles`. Libraries that index blocks with the first axis fastest, such as
n5's `DataBlock`, hold the same memory under the reversed shape, so pass their
block size reversed.

## Speed

Decoding reads the Java arrays in place (JNI critical regions; no copies) and
holds no locks, so threads decode in parallel. On 3×64³ float32 chunks of a
real mouse-template field (ANTs SPIM→STPT, ε = 2.5 µm, 205–238× smaller),
Apple M-series, 14 cores:

| | 1 thread | 8 threads |
|---|---|---|
| Java (`org.warpfield.Bench`) | 800–820 MB/s | 5.1–5.8 GB/s |
| Python, `warpfield_rs` | 796 MB/s | 4.7 GB/s |

Encoding copies its input once (it can take long enough at zstd level 19 that
holding a critical region would stall the garbage collector). Pass a lower zstd
level to `encode` for faster encoding; decoded values do not change.

## Build

Needs Rust (cargo) and a JDK; Maven for the jar.

```sh
java/build_native.sh            # cargo build of rust/warpfield-jni, copy into java/native/, mvn package
java/build_native.sh --no-mvn   # native library only
cd java && mvn test             # tests load rust/target/release directly
```

The library is found, in order, at `-Dwarpfield.library=<file>`, in
`-Dwarpfield.library.dir=<dir>`, then inside the jar at
`org/warpfield/native/<os>-<arch>/`. A local build holds only this machine's
library. `.github/workflows/java.yml` builds all five platforms (Linux via
`cargo zigbuild` for glibc 2.17), tests each on Java 8 and 21, and assembles
one jar with all of them.

## Not here yet

Codec adapters for n5-zarr and zarr-java. Each is a thin class that maps that
library's chunk/block type onto `Warpfield.decode` / `encode`.
