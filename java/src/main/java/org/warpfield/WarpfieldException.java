package org.warpfield;

/** A warpfield encode or decode failed: bad configuration, shape, buffer, or corrupt data. */
public class WarpfieldException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  public WarpfieldException(String message) {
    super(message);
  }
}
