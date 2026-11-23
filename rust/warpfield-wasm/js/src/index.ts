/**
 * warpfield codec for JavaScript/TypeScript, backed by the Rust implementation
 * compiled to WebAssembly. Bit-identical to the Python and native Rust codecs.
 *
 *   const wf = await WarpField.instantiate(wasmBytesOrResponse);
 *   const field = wf.decode(chunkBytes, [64, 64, 64, 3], "float32", config);
 *
 * `config` is the codec's `configuration` object from the zarr array metadata
 * (or the whole `{name: "warpfield", configuration}` entry).
 *
 * The module has no imports and uses no JS glue beyond this file. One instance
 * is single-threaded; use one instance per worker.
 */

export type AxisRole = "spatial" | "smooth" | "independent" | "component";
export type DataType = "float32" | "float64";

export interface JacobianConfig {
  grid_spacing: readonly number[];
  min_det?: number;
  tight_error_bound?: number | number[] | null;
}

export interface WarpFieldConfig {
  axis_roles: readonly AxisRole[];
  error_bound?: number;
  face_error_bound?: number | null;
  predictor?: "linear" | "cubic" | "eno";
  boundary?: "clamped" | "extrapolate";
  jacobian?: JacobianConfig | null;
}

export type ConfigLike =
  | WarpFieldConfig
  | { name: "warpfield"; configuration: WarpFieldConfig }
  | string;

interface Exports {
  memory: WebAssembly.Memory;
  wf_alloc(n: number): number;
  wf_free(p: number, n: number): void;
  wf_error_ptr(): number;
  wf_error_len(): number;
  wf_result_ptr(): number;
  wf_result_len(): number;
  wf_result_clear(): void;
  wf_decode(
    cfgPtr: number,
    cfgLen: number,
    bufPtr: number,
    bufLen: number,
    shapePtr: number,
    rank: number,
    dtype: number,
    outPtr: number,
  ): number;
  wf_encode(
    cfgPtr: number,
    cfgLen: number,
    dataPtr: number,
    shapePtr: number,
    rank: number,
    dtype: number,
    zstdLevel: number,
    alpha: number,
    beta: number,
    effort: number,
  ): number;
}

/** Encoder effort: "fast" (default) tunes on the chunk's central 33^d; "max" on 65^d. */
export type Effort = "fast" | "max";

export class WarpFieldError extends Error {}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

type WasmSource =
  | BufferSource
  | WebAssembly.Module
  | Response
  | PromiseLike<Response>;

export class WarpField {
  readonly exports: Exports;

  constructor(instance: WebAssembly.Instance) {
    this.exports = instance.exports as unknown as Exports;
  }

  /** Instantiate from wasm bytes, a compiled module, or a fetch() response. */
  static async instantiate(source: WasmSource): Promise<WarpField> {
    let module: WebAssembly.Module;
    if (source instanceof WebAssembly.Module) {
      module = source;
    } else if (
      typeof (source as PromiseLike<Response>).then === "function" ||
      (typeof Response !== "undefined" && source instanceof Response)
    ) {
      const resp = await (source as PromiseLike<Response>);
      module = await WebAssembly.compile(await resp.arrayBuffer());
    } else {
      module = await WebAssembly.compile(source as BufferSource);
    }
    return new WarpField(await WebAssembly.instantiate(module, {}));
  }

  private bytes(): Uint8Array {
    return new Uint8Array(this.exports.memory.buffer);
  }

  private put(data: Uint8Array): [number, number] {
    const p = this.exports.wf_alloc(data.byteLength);
    this.bytes().set(data, p); // view taken after alloc: memory may have grown
    return [p, data.byteLength];
  }

  private fail(): never {
    const p = this.exports.wf_error_ptr();
    const n = this.exports.wf_error_len();
    throw new WarpFieldError(decoder.decode(this.bytes().slice(p, p + n)));
  }

  private configBytes(config: ConfigLike): Uint8Array {
    return encoder.encode(
      typeof config === "string" ? config : JSON.stringify(config),
    );
  }

  private shapeBytes(shape: readonly number[]): Uint8Array {
    return new Uint8Array(Uint32Array.from(shape).buffer);
  }

  /** Decode one chunk. Returns a new typed array of prod(shape) values, C order. */
  decode(
    chunk: Uint8Array,
    shape: readonly number[],
    dataType: DataType,
    config: ConfigLike,
  ): Float32Array | Float64Array {
    const n = shape.reduce((a, b) => a * b, 1);
    const size = dataType === "float32" ? 4 : 8;
    const allocs: [number, number][] = [];
    try {
      const cfg = this.put(this.configBytes(config));
      allocs.push(cfg);
      const buf = this.put(chunk);
      allocs.push(buf);
      const shp = this.put(this.shapeBytes(shape));
      allocs.push(shp);
      const outLen = n * size;
      const out = this.exports.wf_alloc(outLen);
      allocs.push([out, outLen]);
      const status = this.exports.wf_decode(
        cfg[0],
        cfg[1],
        buf[0],
        buf[1],
        shp[0],
        shape.length,
        dataType === "float32" ? 0 : 1,
        out,
      );
      if (status !== 0) this.fail();
      const mem = this.exports.memory.buffer;
      return dataType === "float32"
        ? new Float32Array(mem.slice(out, out + outLen))
        : new Float64Array(mem.slice(out, out + outLen));
    } finally {
      for (const [p, len] of allocs) this.exports.wf_free(p, len);
    }
  }

  /**
   * Encode one chunk (C order). `zstdLevel` is ignored by the pure-Rust zstd in wasm.
   * `schedule` fixes the level schedule [alpha, beta] (both >= 1; the level with
   * interpolation step 2^L is coded at error_bound / min(alpha^L, beta)); by default
   * it is tuned per chunk, on a sample whose size `effort` sets.
   */
  encode(
    data: Float32Array | Float64Array,
    shape: readonly number[],
    config: ConfigLike,
    zstdLevel = 3,
    schedule?: readonly [number, number],
    effort: Effort = "fast",
  ): Uint8Array {
    if (effort !== "fast" && effort !== "max") {
      throw new WarpFieldError(`effort must be "fast" or "max", not ${effort}`);
    }
    const n = shape.reduce((a, b) => a * b, 1);
    if (data.length !== n) {
      throw new WarpFieldError(`data has ${data.length} values, shape needs ${n}`);
    }
    const allocs: [number, number][] = [];
    try {
      const cfg = this.put(this.configBytes(config));
      allocs.push(cfg);
      const raw = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const d = this.put(raw);
      allocs.push(d);
      const shp = this.put(this.shapeBytes(shape));
      allocs.push(shp);
      const status = this.exports.wf_encode(
        cfg[0],
        cfg[1],
        d[0],
        shp[0],
        shape.length,
        data instanceof Float32Array ? 0 : 1,
        zstdLevel,
        schedule ? schedule[0] : 0,
        schedule ? schedule[1] : 0,
        effort === "max" ? 1 : 0,
      );
      if (status !== 0) this.fail();
      const p = this.exports.wf_result_ptr();
      const len = this.exports.wf_result_len();
      const out = this.bytes().slice(p, p + len);
      this.exports.wf_result_clear();
      return out;
    } finally {
      for (const [p, len] of allocs) this.exports.wf_free(p, len);
    }
  }
}
