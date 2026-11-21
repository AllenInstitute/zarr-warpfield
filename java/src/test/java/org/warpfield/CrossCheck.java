package org.warpfield;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

/**
 * Cross-check against fixtures written by tests/test_java.py:
 *
 * <pre>java -cp ... org.warpfield.CrossCheck &lt;fixture dir&gt;</pre>
 *
 * <p>Each case is {@code <name>.meta} (dtype, shape, configuration JSON on three lines), {@code
 * <name>.raw} (the original chunk, little-endian) and {@code <name>.encoded} (Python's
 * encoding). Writes {@code <name>.java_decoded} (Python's encoding decoded here) and {@code
 * <name>.java_encoded} (the original encoded here), both little-endian; Python compares.
 */
public final class CrossCheck {
  private CrossCheck() {}

  public static void main(String[] args) throws IOException {
    File dir = new File(args[0]);
    File[] metas = dir.listFiles((d, f) -> f.endsWith(".meta"));
    if (metas == null) {
      throw new IOException("not a directory: " + dir);
    }
    int count = 0;
    for (File meta : metas) {
      String name = meta.getName().substring(0, meta.getName().length() - 5);
      List<String> lines = Files.readAllLines(meta.toPath(), StandardCharsets.UTF_8);
      String dtype = lines.get(0).trim();
      String[] parts = lines.get(1).trim().split("\\s+");
      int[] shape = new int[parts.length];
      for (int i = 0; i < parts.length; i++) {
        shape[i] = Integer.parseInt(parts[i]);
      }
      String config = lines.get(2);
      byte[] encoded = Files.readAllBytes(new File(dir, name + ".encoded").toPath());
      ByteBuffer raw =
          ByteBuffer.wrap(Files.readAllBytes(new File(dir, name + ".raw").toPath()))
              .order(ByteOrder.LITTLE_ENDIAN);
      int n = Warpfield.elementCount(shape);
      byte[] decodedBytes;
      byte[] reencoded;
      if (dtype.equals("float32")) {
        float[] dec = Warpfield.decodeFloat32(encoded, shape, config);
        ByteBuffer b = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(dec);
        decodedBytes = b.array();
        float[] data = new float[n];
        raw.asFloatBuffer().get(data);
        reencoded = Warpfield.encode(data, shape, config);
      } else if (dtype.equals("float64")) {
        double[] dec = Warpfield.decodeFloat64(encoded, shape, config);
        ByteBuffer b = ByteBuffer.allocate(n * 8).order(ByteOrder.LITTLE_ENDIAN);
        b.asDoubleBuffer().put(dec);
        decodedBytes = b.array();
        double[] data = new double[n];
        raw.asDoubleBuffer().get(data);
        reencoded = Warpfield.encode(data, shape, config);
      } else {
        throw new IOException("unknown dtype " + dtype);
      }
      Files.write(new File(dir, name + ".java_decoded").toPath(), decodedBytes);
      Files.write(new File(dir, name + ".java_encoded").toPath(), reencoded);
      count++;
    }
    System.out.println("crosschecked " + count + " cases");
  }
}
