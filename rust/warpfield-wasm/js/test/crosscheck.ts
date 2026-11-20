// Cross-check against fixtures written by tests/test_wasm.py:
//   node test/crosscheck.ts <fixture dir> <warpfield.wasm>
// For each case: decode the Python-encoded chunk (written as <case>.wasm_decoded)
// and encode the original (written as <case>.wasm_encoded); Python compares.
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { WarpField } from "../src/index.ts";

const [dir, wasmPath] = process.argv.slice(2);
const wf = await WarpField.instantiate(readFileSync(wasmPath));
let n = 0;
for (const f of readdirSync(dir).filter((f) => f.endsWith(".json"))) {
  const name = f.slice(0, -5);
  const meta = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const enc = new Uint8Array(readFileSync(join(dir, `${name}.encoded`)));
  const dec = wf.decode(enc, meta.shape, meta.dtype, meta.config);
  writeFileSync(join(dir, `${name}.wasm_decoded`), new Uint8Array(dec.buffer));
  const raw = readFileSync(join(dir, `${name}.raw`));
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const data = meta.dtype === "float32" ? new Float32Array(ab) : new Float64Array(ab);
  writeFileSync(join(dir, `${name}.wasm_encoded`), wf.encode(data, meta.shape, meta.config));
  n++;
}
console.log(`crosschecked ${n} cases`);
