// Load a Neuroglancer state in headless Chromium, wait until every visible
// chunk is loaded, report errors and timing, and save a screenshot.
//   node shot.mjs state.json out.png [timeoutSeconds]
import { readFileSync } from "fs";
import { chromium } from "playwright";
const [stateFile, out, timeoutS = "120"] = process.argv.slice(2);
const state = JSON.parse(readFileSync(stateFile, "utf8"));
const b = await chromium.launch({ args: ["--use-gl=angle", "--enable-unsafe-swiftshader"] });
const p = await b.newPage({ viewport: { width: 1400, height: 900 } });
const errs = [];
p.on("pageerror", (e) => errs.push(String(e.message).slice(0, 400)));
p.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errs.push(m.type() + ": " + m.text().slice(0, 400)); });
const t0 = Date.now();
await p.goto(`http://localhost:8097/#!${encodeURIComponent(JSON.stringify(state))}`, { waitUntil: "domcontentloaded" });
await p.waitForFunction("window.viewer !== undefined", null, { timeout: 60000 });
let ready = false;
try {
  await p.waitForFunction(() => {
    const v = window.viewer;
    const ls = v.layerManager.managedLayers;
    if (!ls.every((l) => l.layer && l.layer.dataSources.every((d) => d.loadState !== undefined))) return false;
    return v.isReady();
  }, null, { timeout: Number(timeoutS) * 1000, polling: 100 });
  ready = true;
} catch {}
const tReady = Date.now() - t0;
const info = await p.evaluate(() => {
  const v = window.viewer;
  return {
    layers: v.layerManager.managedLayers.map((l) => ({
      name: l.name,
      errors: (l.layer?.dataSources ?? []).map((d) => d.loadState?.error ? String(d.loadState.error) : null),
      messages: (l.layer?.dataSources ?? []).flatMap((d) => [...(d.messages?.messages ?? [])].map((m) => m.message)),
    })),
    dims: v.coordinateSpace.value.names,
    position: Array.from(v.position.value),
  };
});
console.log(JSON.stringify({ ready, ms: tReady, ...info }, null, 1));
console.log("errors:", errs.length ? errs.slice(0, 8) : "(none)");
await p.screenshot({ path: out });
await b.close();
