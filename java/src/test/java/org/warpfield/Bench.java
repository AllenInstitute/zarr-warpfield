package org.warpfield;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Decode throughput on fixtures in the {@link CrossCheck} layout ({@code .meta} + {@code
 * .encoded}):
 *
 * <pre>java -cp ... org.warpfield.Bench &lt;fixture dir&gt; [threads]</pre>
 *
 * <p>Reports MB/s of decoded output, single-threaded and with {@code threads} threads.
 */
public final class Bench {
  private Bench() {}

  static final class Case {
    final String name;
    final String dtype;
    final int[] shape;
    final String config;
    final byte[] encoded;

    Case(File dir, File meta) throws IOException {
      name = meta.getName().substring(0, meta.getName().length() - 5);
      List<String> lines = Files.readAllLines(meta.toPath(), StandardCharsets.UTF_8);
      dtype = lines.get(0).trim();
      String[] parts = lines.get(1).trim().split("\\s+");
      shape = new int[parts.length];
      for (int i = 0; i < parts.length; i++) {
        shape[i] = Integer.parseInt(parts[i]);
      }
      config = lines.get(2);
      encoded = Files.readAllBytes(new File(dir, name + ".encoded").toPath());
    }

    long bytes() {
      return (long) Warpfield.elementCount(shape) * (dtype.equals("float32") ? 4 : 8);
    }

    void decode(float[] f, double[] d) {
      if (f != null) {
        Warpfield.decode(encoded, 0, encoded.length, shape, config, f, 0);
      } else {
        Warpfield.decode(encoded, 0, encoded.length, shape, config, d, 0);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    File dir = new File(args[0]);
    int threads = args.length > 1 ? Integer.parseInt(args[1]) : 8;
    File[] metas = dir.listFiles((d, f) -> f.endsWith(".meta"));
    List<Case> cases = new ArrayList<>();
    for (File m : metas) {
      cases.add(new Case(dir, m));
    }
    cases.sort((a, b) -> a.name.compareTo(b.name));
    System.out.printf(
        "%-28s %10s %8s %12s %14s%n", "case", "MB", "ratio", "1 thread", threads + " threads");
    for (Case c : cases) {
      int n = Warpfield.elementCount(c.shape);
      boolean f32 = c.dtype.equals("float32");
      float[] f = f32 ? new float[n] : null;
      double[] d = f32 ? null : new double[n];
      for (int i = 0; i < 5; i++) {
        c.decode(f, d); // warm up
      }
      int reps = (int) Math.max(5, Math.min(200, 2e9 / Math.max(1, c.bytes()) / 10));
      long t0 = System.nanoTime();
      for (int i = 0; i < reps; i++) {
        c.decode(f, d);
      }
      double single = c.bytes() * reps / ((System.nanoTime() - t0) / 1e9) / 1e6;

      ExecutorService pool = Executors.newFixedThreadPool(threads);
      List<Future<?>> fs = new ArrayList<>();
      int per = Math.max(reps, 10);
      long t1 = System.nanoTime();
      for (int t = 0; t < threads; t++) {
        fs.add(
            pool.submit(
                () -> {
                  float[] ff = f32 ? new float[n] : null;
                  double[] dd = f32 ? null : new double[n];
                  for (int i = 0; i < per; i++) {
                    c.decode(ff, dd);
                  }
                }));
      }
      for (Future<?> x : fs) {
        x.get();
      }
      double multi = c.bytes() * (double) per * threads / ((System.nanoTime() - t1) / 1e9) / 1e6;
      pool.shutdown();
      System.out.printf(
          "%-28s %10.2f %8.1f %9.0f MB/s %11.0f MB/s%n",
          c.name, c.bytes() / 1e6, (double) c.bytes() / c.encoded.length, single, multi);
    }
  }
}
