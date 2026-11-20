// Throughput of the wasm codec: node test/bench.ts <dir> <warpfield.wasm>
// <dir> holds <preset>.json ({shape, dtype, config, n}) and <preset>.<i>.encoded /
// <preset>.<i>.raw chunk files written by scripts/throughput.py. Prints one JSON
// line per preset with total decode and encode seconds.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { WarpField } from "../src/index.ts";

const [dir, wasmPath] = process.argv.slice(2);
const wf = await WarpField.instantiate(readFileSync(wasmPath));
for (const f of readdirSync(dir).filter((f) => f.endsWith(".json"))) {
  const name = f.slice(0, -5);
  const meta = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const enc: Uint8Array[] = [];
  const raw: Float32Array[] = [];
  for (let i = 0; i < meta.n; i++) {
    enc.push(new Uint8Array(readFileSync(join(dir, `${name}.${i}.encoded`))));
    const r = readFileSync(join(dir, `${name}.${i}.raw`));
    raw.push(new Float32Array(r.buffer.slice(r.byteOffset, r.byteOffset + r.byteLength)));
  }
  for (const b of enc.slice(0, 2)) wf.decode(b, meta.shape, meta.dtype, meta.config); // warm up
  let t0 = performance.now();
  for (const b of enc) wf.decode(b, meta.shape, meta.dtype, meta.config);
  const dec = (performance.now() - t0) / 1000;
  t0 = performance.now();
  for (const r of raw) wf.encode(r, meta.shape, meta.config);
  const encS = (performance.now() - t0) / 1000;
  console.log(JSON.stringify({ preset: name, decode_s: dec, encode_s: encS }));
}
