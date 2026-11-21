// Shift the SPIM-native layer by the mean displacement (output mm, z y x) and save states.
import { readFileSync, writeFileSync } from "fs";
import { chromium } from "playwright";
const [inFile, outFile, dz, dy, dx] = process.argv.slice(2);
const state = JSON.parse(readFileSync(inFile, "utf8"));
const b = await chromium.launch({ args: ["--use-gl=angle", "--enable-unsafe-swiftshader"] });
const p = await b.newPage();
await p.goto(`http://localhost:8097/#!${encodeURIComponent(JSON.stringify(state))}`);
await p.waitForFunction(() => window.viewer && window.viewer.layerManager.managedLayers[1]?.layer?.dataSources[0]?.loadState?.transform);
const json = await p.evaluate(([dz, dy, dx]) => {
  const v = window.viewer;
  const t = v.layerManager.managedLayers[1].layer.dataSources[0].loadState.transform;
  const val = t.value;
  const r = val.rank;
  const m = Float64Array.from(val.transform);
  const names = val.outputSpace.names;
  const d = { z: dz, y: dy, x: dx };
  for (let i = 0; i < r; ++i) m[r * (r + 1) + i] += (d[names[i]] * 1e-3) / val.outputSpace.scales[i];
  t.value = { ...val, transform: m };
  return v.state.toJSON();
}, [Number(dz), Number(dy), Number(dx)]);
writeFileSync(outFile, JSON.stringify(json));
await b.close();
