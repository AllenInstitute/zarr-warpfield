package org.warpfield;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Finds and loads the native library ({@code warpfield_jni}).
 *
 * <p>Looked for, in order:
 *
 * <ol>
 *   <li>the file named by the system property {@code warpfield.library};
 *   <li>the platform's file name ({@code libwarpfield_jni.dylib}, {@code
 *       libwarpfield_jni.so}, {@code warpfield_jni.dll}) in the directory named by {@code
 *       warpfield.library.dir};
 *   <li>the copy packaged in this jar at {@code /org/warpfield/native/<os>-<arch>/}, extracted
 *       to a temporary file.
 * </ol>
 */
final class NativeLoader {
  static final String LIBRARY = "warpfield_jni";

  private NativeLoader() {}

  /** "macos", "linux" or "windows", then "-", then "x86_64" or "aarch64". */
  static String platform() {
    String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
    String o;
    if (os.startsWith("mac") || os.startsWith("darwin")) {
      o = "macos";
    } else if (os.startsWith("windows")) {
      o = "windows";
    } else if (os.startsWith("linux")) {
      o = "linux";
    } else {
      o = os.replaceAll("[^a-z0-9]", "");
    }
    String a;
    if (arch.equals("amd64") || arch.equals("x86_64") || arch.equals("x64")) {
      a = "x86_64";
    } else if (arch.equals("aarch64") || arch.equals("arm64")) {
      a = "aarch64";
    } else {
      a = arch.replaceAll("[^a-z0-9_]", "");
    }
    return o + "-" + a;
  }

  static void load() {
    String file = System.getProperty("warpfield.library");
    if (file != null && !file.isEmpty()) {
      System.load(new File(file).getAbsolutePath());
      return;
    }
    String name = System.mapLibraryName(LIBRARY);
    String dir = System.getProperty("warpfield.library.dir");
    if (dir != null && !dir.isEmpty()) {
      File f = new File(dir, name);
      if (f.isFile()) {
        System.load(f.getAbsolutePath());
        return;
      }
    }
    String resource = "/org/warpfield/native/" + platform() + "/" + name;
    try (InputStream in = NativeLoader.class.getResourceAsStream(resource)) {
      if (in == null) {
        throw new UnsatisfiedLinkError(
            "warpfield: no native library for "
                + platform()
                + " in this jar ("
                + resource
                + "); build one with java/build_native.sh, or set -Dwarpfield.library=<path>");
      }
      Path tmpDir = Files.createTempDirectory("warpfield-jni");
      Path lib = tmpDir.resolve(name);
      Files.copy(in, lib, StandardCopyOption.REPLACE_EXISTING);
      lib.toFile().deleteOnExit();
      tmpDir.toFile().deleteOnExit();
      System.load(lib.toAbsolutePath().toString());
    } catch (IOException e) {
      UnsatisfiedLinkError err =
          new UnsatisfiedLinkError("warpfield: could not extract " + resource + ": " + e);
      err.initCause(e);
      throw err;
    }
  }
}
