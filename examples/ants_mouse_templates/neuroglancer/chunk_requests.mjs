// Per-URL accounting of template/field chunk requests while a state loads:
// requests, unique URLs, duplicates, bytes; writes the requested chunk keys.
//   node bytes2.mjs state.json out.json
import { readFileSync, writeFileSync } from "fs";
import { chromium } from "playwright";
const [stateFile, outFile] = process.argv.slice(2);
const state = JSON.parse(readFileSync(stateFile, "utf8"));
const b = await chromium.launch({ args: ["--use-gl=angle", "--enable-unsafe-swiftshader"] });
const ctx = await b.newContext({ viewport: { width: 1400, height: 900 } });
const p = await ctx.newPage();
const reqs = [];
const t0 = Date.now();
const classify = (u) => {
  let m = u.match(/(spim|stpt|annotation)[^/]*\/[^/]*\/(?:template|annotations_compressed)\.ome\.zarr\/(s\d)\/c\/(.*)$/);
  if (m) return [`${m[1]} ${m[2]}`, m[3]];
  m = u.match(/annotations\.precomputed\/mesh\/(.*)$/);
  if (m) return [`mesh ${m[1].endsWith(".index") ? "index" : "fragments"}`, m[1]];
  m = u.match(/annotations\.precomputed\/(\d+_\d+_\d+)\/(.*)$/);
  if (m) return [`annot ${m[1]}`, m[2]];
  m = u.match(/127\.0\.0\.1:9077\/[^/]+\/(s\d|bounds)\/c\/(.*)$/);
  if (m) return [`field ${m[1]}`, m[2]];
  return null;
};
ctx.on("requestfinished", async (req) => {
  const c = classify(req.url());
  if (!c) return;
  const sz = (await req.sizes()).responseBodySize;
  if (c[0].startsWith("mesh")) c[1] += " " + (req.headers()["range"] ?? "");
  reqs.push({ k: c[0], key: c[1], sz, t: Date.now() - t0 });
});
ctx.on("requestfailed", (req) => {
  const c = classify(req.url());
  if (c) reqs.push({ k: c[0], key: c[1], sz: 0, failed: req.failure()?.errorText, t: Date.now() - t0 });
});
await p.goto(`http://localhost:8097/#!${encodeURIComponent(JSON.stringify(state))}`);
await p.waitForFunction("window.viewer !== undefined");
await p.waitForFunction(() => window.viewer.isReady(), null, { timeout: 300000, polling: 100 });
const ready = Date.now() - t0;
await p.waitForTimeout(1500);
const by = {};
for (const r of reqs) {
  const s = (by[r.k] ??= { requests: 0, failed: 0, urls: new Map(), mb: 0 });
  s.requests++; s.mb += r.sz / 1e6;
  if (r.failed) s.failed++;
  s.urls.set(r.key, (s.urls.get(r.key) ?? 0) + 1);
}
const out = {};
for (const [k, s] of Object.entries(by).sort()) {
  const dup = [...s.urls.values()].filter((n) => n > 1).length;
  console.log(k.padEnd(12), `requests ${String(s.requests).padStart(4)}  unique ${String(s.urls.size).padStart(4)}  duplicated ${dup}  cancelled ${s.failed}  ${s.mb.toFixed(1).padStart(7)} MB`);
  out[k] = [...s.urls.keys()];
}
console.log("ready", ready, "ms");
writeFileSync(outFile, JSON.stringify(out));
await b.close();
