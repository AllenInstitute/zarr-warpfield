package org.warpfield;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

class WarpfieldTest {
  // OME-Zarr field layout: (c, z, y, x); no spatial extent is a power of two
  static final int[] SHAPE = {3, 24, 20, 17};
  static final String CONFIG =
      "{\"axis_roles\": [\"component\", \"spatial\", \"spatial\", \"spatial\"],"
          + " \"error_bound\": 0.25}";
  static final String JACOBIAN_CONFIG =
      "{\"axis_roles\": [\"component\", \"spatial\", \"spatial\", \"spatial\"],"
          + " \"error_bound\": 0.1, \"jacobian\": {\"grid_spacing\": [1, 1, 1], \"min_det\": 0.1}}";

  /** A smooth field of a few voxels' displacement plus solver-like noise. */
  static double[] field(int[] shape, long seed) {
    Random rng = new Random(seed);
    int n = Warpfield.elementCount(shape);
    double[] u = new double[n];
    int i = 0;
    for (int c = 0; c < shape[0]; c++) {
      double a = 0.1 + 0.05 * c, b = 0.07 - 0.02 * c, d = 0.05 + 0.03 * c;
      for (int z = 0; z < shape[1]; z++) {
        for (int y = 0; y < shape[2]; y++) {
          for (int x = 0; x < shape[3]; x++) {
            u[i++] = 4.0 * Math.sin(a * z + b * y + d * x + c) + 0.01 * rng.nextGaussian();
          }
        }
      }
    }
    return u;
  }

  static float[] toFloat(double[] d) {
    float[] f = new float[d.length];
    for (int i = 0; i < d.length; i++) {
      f[i] = (float) d[i];
    }
    return f;
  }

  @Test
  void nativeLibraryLoads() {
    assertFalse(Warpfield.nativeVersion().isEmpty());
  }

  @Test
  void float32RoundTripWithinBound() {
    float[] u = toFloat(field(SHAPE, 0));
    byte[] enc = Warpfield.encode(u, SHAPE, CONFIG);
    assertTrue(enc.length < u.length * 4 / 4, "expected at least 4x compression");
    float[] out = Warpfield.decodeFloat32(enc, SHAPE, CONFIG);
    double max = 0;
    for (int i = 0; i < u.length; i++) {
      max = Math.max(max, Math.abs(out[i] - u[i]));
    }
    assertTrue(max <= 0.25, "max error " + max);
  }

  @Test
  void float64RoundTripWithinBoundAndIdempotent() {
    double[] u = field(SHAPE, 1);
    double[] out = Warpfield.decodeFloat64(Warpfield.encode(u, SHAPE, CONFIG), SHAPE, CONFIG);
    for (int i = 0; i < u.length; i++) {
      assertTrue(Math.abs(out[i] - u[i]) <= 0.25);
    }
    // closed-loop prediction: decoded values re-encode to themselves
    double[] again =
        Warpfield.decodeFloat64(Warpfield.encode(out, SHAPE, CONFIG), SHAPE, CONFIG);
    assertArrayEquals(out, again);
  }

  @Test
  void jacobianConfiguration() {
    float[] u = toFloat(field(SHAPE, 2));
    float[] out =
        Warpfield.decodeFloat32(
            Warpfield.encode(u, SHAPE, JACOBIAN_CONFIG), SHAPE, JACOBIAN_CONFIG);
    for (int i = 0; i < u.length; i++) {
      assertTrue(Math.abs(out[i] - u[i]) <= 0.1);
    }
  }

  @Test
  void zstdLevelDoesNotChangeValues() {
    float[] u = toFloat(field(SHAPE, 3));
    byte[] fast = Warpfield.encode(u, 0, SHAPE, CONFIG, 1);
    byte[] small = Warpfield.encode(u, SHAPE, CONFIG);
    assertArrayEquals(
        Warpfield.decodeFloat32(fast, SHAPE, CONFIG),
        Warpfield.decodeFloat32(small, SHAPE, CONFIG));
  }

  @Test
  void offsetsInSourceAndDestination() {
    float[] u = toFloat(field(SHAPE, 4));
    byte[] enc = Warpfield.encode(u, SHAPE, CONFIG);
    float[] ref = Warpfield.decodeFloat32(enc, SHAPE, CONFIG);

    byte[] padded = new byte[enc.length + 13];
    Arrays.fill(padded, (byte) 0x5a);
    System.arraycopy(enc, 0, padded, 7, enc.length);
    float[] dst = new float[ref.length + 11];
    Arrays.fill(dst, -1f);
    Warpfield.decode(padded, 7, enc.length, SHAPE, CONFIG, dst, 5);
    assertArrayEquals(ref, Arrays.copyOfRange(dst, 5, 5 + ref.length));
    for (int i = 0; i < 5; i++) {
      assertEquals(-1f, dst[i]);
    }
    for (int i = 5 + ref.length; i < dst.length; i++) {
      assertEquals(-1f, dst[i]);
    }

    // encoding from an offset
    float[] shifted = new float[u.length + 3];
    System.arraycopy(u, 0, shifted, 3, u.length);
    assertArrayEquals(enc, Warpfield.encode(shifted, 3, SHAPE, CONFIG, 19));
  }

  @Test
  void directBuffersMatchArrays() {
    double[] u = field(SHAPE, 5);
    byte[] enc32 = Warpfield.encode(toFloat(u), SHAPE, CONFIG);
    byte[] enc64 = Warpfield.encode(u, SHAPE, CONFIG);
    int n = u.length;

    ByteBuffer src = ByteBuffer.allocateDirect(enc32.length + 9);
    src.position(4);
    src.put(enc32);
    src.position(4).limit(4 + enc32.length);
    FloatBuffer f =
        ByteBuffer.allocateDirect((n + 2) * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    f.position(2);
    Warpfield.decode(src, SHAPE, CONFIG, f);
    assertEquals(src.limit(), src.position());
    assertEquals(n + 2, f.position());
    float[] got = new float[n];
    f.position(2);
    f.get(got);
    assertArrayEquals(Warpfield.decodeFloat32(enc32, SHAPE, CONFIG), got);

    ByteBuffer src64 = ByteBuffer.allocateDirect(enc64.length);
    src64.put(enc64).flip();
    DoubleBuffer d = ByteBuffer.allocateDirect(n * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
    Warpfield.decode(src64, SHAPE, CONFIG, d);
    double[] got64 = new double[n];
    d.flip();
    d.get(got64);
    assertArrayEquals(Warpfield.decodeFloat64(enc64, SHAPE, CONFIG), got64);
  }

  @Test
  void directBufferMisuseIsRejected() {
    byte[] enc = Warpfield.encode(toFloat(field(SHAPE, 6)), SHAPE, CONFIG);
    int n = Warpfield.elementCount(SHAPE);
    ByteBuffer src = ByteBuffer.allocateDirect(enc.length);
    src.put(enc).flip();
    // heap destination
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decode(src.duplicate(), SHAPE, CONFIG, FloatBuffer.allocate(n)));
    // big-endian view
    FloatBuffer be =
        ByteBuffer.allocateDirect(n * 4).order(ByteOrder.BIG_ENDIAN).asFloatBuffer();
    if (ByteOrder.nativeOrder() != ByteOrder.BIG_ENDIAN) {
      assertThrows(
          WarpfieldException.class, () -> Warpfield.decode(src.duplicate(), SHAPE, CONFIG, be));
    }
    // too small
    FloatBuffer small =
        ByteBuffer.allocateDirect((n - 1) * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    assertThrows(
        WarpfieldException.class, () -> Warpfield.decode(src.duplicate(), SHAPE, CONFIG, small));
    // misaligned view: a float view at byte offset 1
    ByteBuffer raw = ByteBuffer.allocateDirect(n * 4 + 1).order(ByteOrder.nativeOrder());
    raw.position(1);
    FloatBuffer odd = raw.slice().order(ByteOrder.nativeOrder()).asFloatBuffer();
    WarpfieldException e =
        assertThrows(
            WarpfieldException.class, () -> Warpfield.decode(src.duplicate(), SHAPE, CONFIG, odd));
    assertTrue(e.getMessage().contains("aligned"), e.getMessage());
  }

  @Test
  void badInputsThrowInsteadOfCrashing() {
    float[] u = toFloat(field(SHAPE, 7));
    byte[] enc = Warpfield.encode(u, SHAPE, CONFIG);
    float[] dst = new float[u.length];

    assertThrows(
        WarpfieldException.class, () -> Warpfield.decodeFloat32(enc, SHAPE, "{not json"));
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decodeFloat32(enc, SHAPE, "{\"axis_roles\": [\"spatial\"]}"));
    // the shape does not match the encoded chunk
    assertThrows(
        WarpfieldException.class, () -> Warpfield.decodeFloat32(enc, new int[] {3, 24, 20, 16}, CONFIG));
    // truncated, corrupted and empty data
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decodeFloat32(Arrays.copyOf(enc, enc.length / 2), SHAPE, CONFIG));
    byte[] junk = enc.clone();
    for (int i = 16; i < junk.length; i += 7) {
      junk[i] ^= (byte) 0xff;
    }
    assertThrows(WarpfieldException.class, () -> Warpfield.decodeFloat32(junk, SHAPE, CONFIG));
    assertThrows(WarpfieldException.class, () -> Warpfield.decodeFloat32(new byte[0], SHAPE, CONFIG));
    // out-of-range offsets and lengths
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decode(enc, -1, enc.length, SHAPE, CONFIG, dst, 0));
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decode(enc, 1, enc.length, SHAPE, CONFIG, dst, 0));
    assertThrows(
        WarpfieldException.class, () -> Warpfield.decode(enc, 0, enc.length, SHAPE, CONFIG, dst, 1));
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.decode(enc, 0, enc.length, SHAPE, CONFIG, new float[u.length - 1], 0));
    assertThrows(
        WarpfieldException.class, () -> Warpfield.encode(u, 1, SHAPE, CONFIG, 19));
    assertThrows(WarpfieldException.class, () -> Warpfield.decodeFloat32(enc, SHAPE, null));
    // a jacobian configuration needs one component per spatial axis
    assertThrows(
        WarpfieldException.class,
        () -> Warpfield.encode(new float[2 * 24 * 20 * 17], new int[] {2, 24, 20, 17}, JACOBIAN_CONFIG));
    // the library is still usable afterwards
    assertArrayEquals(Warpfield.decodeFloat32(enc, SHAPE, CONFIG), Warpfield.decodeFloat32(enc, SHAPE, CONFIG));
  }

  @Test
  void threadsDecodeInParallel() throws Exception {
    List<byte[]> chunks = new ArrayList<>();
    List<float[]> refs = new ArrayList<>();
    for (int s = 0; s < 4; s++) {
      byte[] enc = Warpfield.encode(toFloat(field(SHAPE, 10 + s)), SHAPE, CONFIG);
      chunks.add(enc);
      refs.add(Warpfield.decodeFloat32(enc, SHAPE, CONFIG));
    }
    ExecutorService pool = Executors.newFixedThreadPool(8);
    try {
      List<Future<Boolean>> results = new ArrayList<>();
      for (int t = 0; t < 64; t++) {
        final int k = t % chunks.size();
        results.add(
            pool.submit(
                () -> Arrays.equals(refs.get(k), Warpfield.decodeFloat32(chunks.get(k), SHAPE, CONFIG))));
      }
      for (Future<Boolean> r : results) {
        assertTrue(r.get());
      }
    } finally {
      pool.shutdown();
    }
  }
}
