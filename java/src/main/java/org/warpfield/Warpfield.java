package org.warpfield;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/**
 * The warpfield codec: error-bounded, seam-free compression of registration displacement
 * fields, one chunk at a time. Calls the Rust implementation through JNI; decoded values are
 * bit-identical to the Python, Rust and WebAssembly decoders.
 *
 * <p><b>Arrays.</b> A chunk is a flat array in C order (last axis fastest) with the given
 * {@code shape}, the same order as the codec's {@code axis_roles}. A zarr chunk of shape {@code
 * [3, 64, 64, 64]} is read with {@code shape = {3, 64, 64, 64}}. Libraries that index blocks
 * with the first axis fastest (n5's {@code DataBlock}) hold the same memory under the reversed
 * shape, so pass their block size reversed.
 *
 * <p><b>Configuration.</b> {@code config} is the codec's {@code configuration} object as JSON,
 * exactly as in the zarr metadata, for example {@code {"axis_roles": ["component", "spatial",
 * "spatial", "spatial"], "error_bound": 0.25}}.
 *
 * <p><b>Speed.</b> Decoding reads Java arrays in place (no copies), and the native code holds
 * no locks, so threads decode in parallel. Encoding copies its input once.
 *
 * <p>All methods throw {@link WarpfieldException} for invalid arguments or corrupt data.
 */
public final class Warpfield {
  /** zstd level used by the reference encoder; the level never changes decoded values. */
  public static final int DEFAULT_ZSTD_LEVEL = 19;

  static {
    NativeLoader.load();
  }

  private Warpfield() {}

  /** Version of the native library. */
  public static String nativeVersion() {
    return nVersion();
  }

  /** Number of elements in a chunk of {@code shape}. */
  public static int elementCount(int[] shape) {
    long n = 1;
    for (int s : shape) {
      if (s < 0) {
        throw new WarpfieldException("negative extent in shape");
      }
      n *= s;
      if (n > Integer.MAX_VALUE) {
        throw new WarpfieldException("chunk too large for a Java array");
      }
    }
    return (int) n;
  }

  // ---- decode -------------------------------------------------------------------------

  /** Decode a float32 chunk. */
  public static float[] decodeFloat32(byte[] encoded, int[] shape, String config) {
    float[] out = new float[elementCount(shape)];
    decode(encoded, 0, encoded.length, shape, config, out, 0);
    return out;
  }

  /** Decode a float64 chunk. */
  public static double[] decodeFloat64(byte[] encoded, int[] shape, String config) {
    double[] out = new double[elementCount(shape)];
    decode(encoded, 0, encoded.length, shape, config, out, 0);
    return out;
  }

  /**
   * Decode {@code src[srcOffset, srcOffset + srcLength)} into {@code dst}, starting at {@code
   * dstOffset} (for example to reuse one output array across chunks).
   */
  public static void decode(
      byte[] src,
      int srcOffset,
      int srcLength,
      int[] shape,
      String config,
      float[] dst,
      int dstOffset) {
    check(src, shape, config, dst);
    nDecodeFloat32(config, src, srcOffset, srcLength, shape, dst, dstOffset);
  }

  /** As {@link #decode(byte[], int, int, int[], String, float[], int)}, for float64. */
  public static void decode(
      byte[] src,
      int srcOffset,
      int srcLength,
      int[] shape,
      String config,
      double[] dst,
      int dstOffset) {
    check(src, shape, config, dst);
    nDecodeFloat64(config, src, srcOffset, srcLength, shape, dst, dstOffset);
  }

  /**
   * Decode the bytes of the direct buffer {@code src} between its position and limit into the
   * direct, native-order buffer {@code dst} from its position. Afterwards {@code src}'s position
   * is its limit and {@code dst}'s position has advanced by the chunk's element count, as for
   * the NIO bulk methods.
   */
  public static void decode(ByteBuffer src, int[] shape, String config, FloatBuffer dst) {
    decodeDirect(src, shape, config, dst, 0, dst.order());
  }

  /** As {@link #decode(ByteBuffer, int[], String, FloatBuffer)}, for float64. */
  public static void decode(ByteBuffer src, int[] shape, String config, DoubleBuffer dst) {
    decodeDirect(src, shape, config, dst, 1, dst.order());
  }

  private static void decodeDirect(
      ByteBuffer src, int[] shape, String config, Buffer dst, int dtype, ByteOrder order) {
    check(src, shape, config, dst);
    if (!src.isDirect() || !dst.isDirect()) {
      throw new WarpfieldException(
          "direct buffers are required (use the array methods for heap buffers)");
    }
    if (order != ByteOrder.nativeOrder()) {
      throw new WarpfieldException("the destination buffer must be in native byte order");
    }
    if (dst.isReadOnly()) {
      throw new WarpfieldException("the destination buffer is read-only");
    }
    int n = elementCount(shape);
    if (dst.remaining() < n) {
      throw new WarpfieldException(
          "the destination has " + dst.remaining() + " elements remaining, the chunk " + n);
    }
    nDecodeDirect(
        config, src, src.position(), src.remaining(), shape, dtype, dst, dst.position());
    src.position(src.limit());
    dst.position(dst.position() + n);
  }

  // ---- encode -------------------------------------------------------------------------

  /** Encode a float32 chunk at the default zstd level. */
  public static byte[] encode(float[] data, int[] shape, String config) {
    return encode(data, 0, shape, config, DEFAULT_ZSTD_LEVEL);
  }

  /** Encode a float64 chunk at the default zstd level. */
  public static byte[] encode(double[] data, int[] shape, String config) {
    return encode(data, 0, shape, config, DEFAULT_ZSTD_LEVEL);
  }

  /**
   * Encode the chunk at {@code data[offset ...]}. Lower zstd levels encode faster and give
   * slightly larger output; decoded values do not change. The level schedule is tuned per
   * chunk.
   */
  public static byte[] encode(
      float[] data, int offset, int[] shape, String config, int zstdLevel) {
    return encode(data, offset, shape, config, zstdLevel, 0, 0);
  }

  /** As {@link #encode(float[], int, int[], String, int)}, for float64. */
  public static byte[] encode(
      double[] data, int offset, int[] shape, String config, int zstdLevel) {
    return encode(data, offset, shape, config, zstdLevel, 0, 0);
  }

  /**
   * Encode with a fixed level schedule: the level with interpolation step 2^L is coded at
   * {@code error_bound / min(alpha^L, beta)}. Both must be at least 1; {@code (1, 1)} is
   * uniform, and {@code (0, 0)} tunes the schedule per chunk (the default). With a fixed
   * schedule, decoded values re-encode to themselves.
   */
  public static byte[] encode(
      float[] data,
      int offset,
      int[] shape,
      String config,
      int zstdLevel,
      double alpha,
      double beta) {
    check(data, shape, config, data);
    return nEncodeFloat32(config, data, offset, shape, zstdLevel, alpha, beta);
  }

  /** As {@link #encode(float[], int, int[], String, int, double, double)}, for float64. */
  public static byte[] encode(
      double[] data,
      int offset,
      int[] shape,
      String config,
      int zstdLevel,
      double alpha,
      double beta) {
    check(data, shape, config, data);
    return nEncodeFloat64(config, data, offset, shape, zstdLevel, alpha, beta);
  }

  private static void check(Object src, int[] shape, String config, Object dst) {
    if (src == null || shape == null || config == null || dst == null) {
      throw new WarpfieldException("null argument");
    }
  }

  // ---- native -------------------------------------------------------------------------

  private static native String nVersion();

  private static native void nDecodeFloat32(
      String config,
      byte[] src,
      int srcOffset,
      int srcLength,
      int[] shape,
      float[] dst,
      int dstOffset);

  private static native void nDecodeFloat64(
      String config,
      byte[] src,
      int srcOffset,
      int srcLength,
      int[] shape,
      double[] dst,
      int dstOffset);

  private static native void nDecodeDirect(
      String config,
      ByteBuffer src,
      int srcPosition,
      int srcLength,
      int[] shape,
      int dtype,
      Buffer dst,
      int dstPosition);

  private static native byte[] nEncodeFloat32(
      String config,
      float[] data,
      int offset,
      int[] shape,
      int zstdLevel,
      double alpha,
      double beta);

  private static native byte[] nEncodeFloat64(
      String config,
      double[] data,
      int offset,
      int[] shape,
      int zstdLevel,
      double alpha,
      double beta);
}
