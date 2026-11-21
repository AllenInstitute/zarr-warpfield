// Displacement-field shader for a warpfield OME-Zarr image layer whose vector
// axis is a channel dimension ("c^"). Shows either
//  - det J of x -> x + u(x), computed here from neighbouring voxels of the
//    chunk (central differences inside the chunk, one-sided on chunk faces):
//    blue = compression, white = volume preserved, red = expansion,
//    yellow = det J <= min_det (near or past a fold); or
//  - the displacement minus a constant offset, as signed RGB
//    (gray = no residual displacement).
#uicontrol bool show_det_j checkbox(default=true)
#uicontrol float log2_range slider(min=0.1, max=3.0, default=1.0)
#uicontrol float min_det slider(min=0.0, max=1.0, default=0.1)
#uicontrol float spacing_mm slider(min=0.001, max=0.2, default=0.025)
#uicontrol float range_mm slider(min=0.05, max=2.0, default=0.5)
#uicontrol float offset0 slider(min=-10, max=10, default=5.2394)
#uicontrol float offset1 slider(min=-10, max=10, default=5.2192)
#uicontrol float offset2 slider(min=-10, max=10, default=-5.6828)

// getDataValueAt(p, c) indexes the chunk in its own axes, whatever the view
// orientation: for a (c, k, j, i) zarr array, p = (i, j, k). Component c is the
// displacement along array axis c of (k, j, i), i.e. along p[2 - c].

// d u_c / d p[b], per mm (field sample spacing spacing_mm).
float du(int c, int b) {
  ivec3 size = ivec3(uChunkDataSize);
  ivec3 p = clamp(ivec3(floor(vChunkPosition)), ivec3(0), size - 1);
  ivec3 e = ivec3(0);
  e[b] = 1;
  ivec3 lo = max(p - e, ivec3(0));
  ivec3 hi = min(p + e, size - 1);
  int n = hi[b] - lo[b];
  if (n == 0) return 0.0;
  return (getDataValueAt(hi, c) - getDataValueAt(lo, c)) / (float(n) * spacing_mm);
}

void main() {
  if (!show_det_j) {
    vec3 u = vec3(getDataValue(0), getDataValue(1), getDataValue(2)) -
             vec3(offset0, offset1, offset2);
    emitRGB(clamp(0.5 + 0.5 * u / range_mm, 0.0, 1.0));
    return;
  }
  mat3 J;  // J[col][row]: row = component a, col = array axis b
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      J[b][a] = (a == b ? 1.0 : 0.0) + du(a, 2 - b);
    }
  }
  float d = determinant(J);
  if (d <= min_det) {
    emitRGB(vec3(1.0, 0.9, 0.0));
    return;
  }
  float t = clamp(log2(d) / log2_range, -1.0, 1.0);
  emitRGB(t < 0.0 ? mix(vec3(1.0), vec3(0.15, 0.35, 0.9), -t)
                  : mix(vec3(1.0), vec3(0.85, 0.15, 0.1), t));
}
