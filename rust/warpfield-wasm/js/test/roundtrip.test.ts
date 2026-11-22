import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { WarpField, WarpFieldError } from "../src/index.ts";

const wasm = readFileSync(new URL("../warpfield.wasm", import.meta.url));
const wf = await WarpField.instantiate(wasm);

// A smooth 2D field with a 6-voxel slip crack across it.
function field(ny: number, nx: number): Float32Array {
  const a = new Float32Array(ny * nx * 2);
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++) {
      const i = (y * nx + x) * 2;
      const jump = x + 0.3 * y > 40 ? 6 : 0;
      a[i] = 3 * Math.sin(y / 17) + 0.05 * x + jump;
      a[i + 1] = 2 * Math.cos(x / 23) - 0.02 * y;
    }
  return a;
}

const config = {
  axis_roles: ["spatial", "spatial", "component"],
  error_bound: 0.25,
  jacobian: { grid_spacing: [4, 4], min_det: 0.1 },
} as const;

test("round trip respects the error bounds", () => {
  const shape = [70, 90, 2];
  const x = field(70, 90);
  const buf = wf.encode(x, shape, { name: "warpfield", configuration: config });
  assert.ok(buf.length < x.byteLength / 10, `ratio ${x.byteLength / buf.length}`);
  const y = wf.decode(buf, shape, "float32", config);
  let maxInterior = 0;
  let maxFace = 0;
  for (let yy = 0; yy < 70; yy++)
    for (let xx = 0; xx < 90; xx++)
      for (let c = 0; c < 2; c++) {
        const i = (yy * 90 + xx) * 2 + c;
        const e = Math.abs(x[i] - y[i]);
        const face = yy === 0 || yy === 69 || xx === 0 || xx === 89;
        if (face) maxFace = Math.max(maxFace, e);
        else maxInterior = Math.max(maxInterior, e);
      }
  assert.ok(maxInterior <= 0.25, `interior ${maxInterior}`);
  assert.ok(maxFace <= 0.0625, `face ${maxFace}`);
});

test("float64 round trip", () => {
  const shape = [33, 20, 2];
  const x = Float64Array.from(field(33, 20));
  const y = wf.decode(wf.encode(x, shape, config), shape, "float64", config);
  assert.ok(y instanceof Float64Array);
  for (let i = 0; i < x.length; i++) assert.ok(Math.abs(x[i] - y[i]) <= 0.25);
});

test("a fixed level schedule re-encodes decoded values to themselves", () => {
  const shape = [33, 20, 2];
  const x = Float32Array.from(field(33, 20));
  const once = wf.decode(wf.encode(x, shape, config, 3, [1.1, 1.5]), shape, "float32", config);
  const twice = wf.decode(
    wf.encode(once, shape, config, 3, [1.1, 1.5]),
    shape,
    "float32",
    config,
  );
  assert.deepEqual(Array.from(twice), Array.from(once));
  assert.throws(() => wf.encode(x, shape, config, 3, [0.5, 2]), WarpFieldError);
});

test("errors are reported, not thrown as traps", () => {
  const shape = [16, 16, 2];
  const buf = wf.encode(field(16, 16), shape, config);
  assert.throws(() => wf.decode(buf.slice(0, 20), shape, "float32", config), WarpFieldError);
  assert.throws(() => wf.decode(buf, [16, 17, 2], "float32", config), /canonical shape/);
  assert.throws(
    () => wf.decode(buf, shape, "float32", { axis_roles: ["spatial"] }),
    /rank/,
  );
  // still usable afterwards
  assert.equal(wf.decode(buf, shape, "float32", config).length, 512);
});
