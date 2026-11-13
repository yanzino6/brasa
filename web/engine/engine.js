// Hand-written inference engine for Brasa (the model defined in src/brasa/model.py). No dependencies.
// Runs in a browser page, in a Web Worker and in Node.
//
// Everything numeric lives in ONE Float32Array "heap" (weights, KV cache, scratch vectors) addressed by
// float offsets. That lets the two heavy loops (matrix x vector, weighted row sum) be swapped between
//   * a plain JavaScript kernel, and
//   * a 1.5 kB WebAssembly SIMD kernel (../wasm/kernel.c) working on the same memory.
//
// One token is processed per step(): its keys and values are appended to the cache, so generating a
// token costs a single-position forward pass. Nothing is allocated while generating.
//
// What a visualisation needs is available without slowing the default path:
//   topTokens(logits, k)            the k most likely next tokens with their probabilities
//   setCapture({ attention, norms }) then attention() / residualNorms() after any step
//   stream(prompt, { inspect })     the same, attached to every generated token
//
// Model file ("TGPT" container, little endian), written by src/brasa/export.py:
//   "TGPT" | u32 version | u32 headerBytes | header JSON (space padded to 4 bytes) | tensor data
//   header = { format, name, config, prompt, tokenizer,
//              tensors: [{ name, shape, dtype: "f32"|"f16"|"i8", offset, scalesOffset? }] }
//   int8 matrices are expanded to float32 at load time: w = q * scales[row].

const MAGIC = 0x54504754; // "TGPT"
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** Split a model file into its JSON header and the position of the tensor data. */
export function parseModel(buffer) {
  if (ArrayBuffer.isView(buffer)) {
    buffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  }
  const dv = new DataView(buffer);
  if (buffer.byteLength < 12 || dv.getUint32(0, true) !== MAGIC) throw new Error('not a TGPT model file');
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported TGPT version ${version}`);
  const headerBytes = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, headerBytes)));
  return { header, dataStart: 12 + headerBytes, buffer };
}

// ------------------------------------------------------------------------------------------ kernels

function jsKernel(heap) {
  return {
    name: 'js',
    // y[r] = sum_c W[r, c] * x[c]
    matvec(w, x, y, rows, cols) {
      const c4 = cols & ~3;
      for (let r = 0; r < rows; r++) {
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
        for (; c < c4; c += 4) {
          s0 += heap[w + c] * heap[x + c];
          s1 += heap[w + c + 1] * heap[x + c + 1];
          s2 += heap[w + c + 2] * heap[x + c + 2];
          s3 += heap[w + c + 3] * heap[x + c + 3];
        }
        for (; c < cols; c++) s0 += heap[w + c] * heap[x + c];
        heap[y + r] = s0 + s1 + s2 + s3;
        w += cols;
      }
    },
    // out[c] = sum_r a[r] * V[r, c]
    tmatvec(v, a, out, rows, cols) {
      for (let c = 0; c < cols; c++) heap[out + c] = 0;
      for (let r = 0; r < rows; r++) {
        const ar = heap[a + r];
        for (let c = 0; c < cols; c++) heap[out + c] += ar * heap[v + c];
        v += cols;
      }
    },
  };
}

function wasmKernel(instance) {
  const { matvec, tmatvec } = instance.exports;
  return {
    name: 'wasm-simd',
    matvec: (w, x, y, rows, cols) => matvec(w << 2, x << 2, y << 2, rows, cols),
    tmatvec: (v, a, out, rows, cols) => tmatvec(v << 2, a << 2, out << 2, rows, cols),
  };
}

let f16Table = null;
function halfTable() { // IEEE 754 binary16 -> float32, for all 65536 bit patterns
  if (!f16Table) {
    f16Table = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) {
      const s = i & 0x8000 ? -1 : 1, e = (i >> 10) & 0x1f, m = i & 0x3ff;
      f16Table[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
    }
  }
  return f16Table;
}

// ------------------------------------------------------------------------------------------ sampling

/** A small seedable random number generator (mulberry32), for reproducible sampling. */
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function argmax(logits) {
  let best = 0, bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}

/**
 * The k most likely tokens under softmax(logits), most likely first: [{ id, prob, logit }].
 * This is the model's own distribution (temperature 1, nothing filtered).
 */
export function topTokens(logits, k = 5) {
  const V = logits.length;
  k = Math.min(k, V);
  const ids = new Int32Array(k).fill(-1);
  let max = -Infinity, filled = 0;
  for (let i = 0; i < V; i++) {
    const v = logits[i];
    if (v > max) max = v;
    if (filled < k || v > logits[ids[k - 1]]) {   // insert into the short sorted list
      let j = filled < k ? filled++ : k - 1;
      for (; j > 0 && logits[ids[j - 1]] < v; j--) ids[j] = ids[j - 1];
      ids[j] = i;
    }
  }
  let sum = 0;
  for (let i = 0; i < V; i++) sum += Math.exp(logits[i] - max);
  return Array.from(ids, (id) => ({ id, prob: Math.exp(logits[id] - max) / sum, logit: logits[id] }));
}

/** softmax(logits)[id] */
export function probabilityOf(logits, id) {
  let max = -Infinity, sum = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return Math.exp(logits[id] - max) / sum;
}

function penalized(logits, repetitionPenalty, previous) {
  const z = Float64Array.from(logits);
  if (repetitionPenalty !== 1 && previous && previous.length) {
    for (const id of new Set(previous)) z[id] = z[id] > 0 ? z[id] / repetitionPenalty : z[id] * repetitionPenalty;
  }
  return z;
}

/**
 * The distribution a token is sampled from. Same order of operations as GPT.filter_logits in model.py:
 *   1. repetition penalty on the tokens in `previous` (positive logits divided, negative multiplied)
 *   2. temperature
 *   3. top-k: keep the k largest (ties with the k-th are kept)
 *   4. top-p: keep the smallest prefix of the sorted distribution whose mass reaches topP
 * Returns the kept token ids, most likely first, and their renormalised probabilities.
 *
 * @returns {{ ids: Int32Array, probs: Float64Array }}
 */
export function filterLogits(logits, { temperature = 1, topK = 0, topP = 1, repetitionPenalty = 1 } = {}, previous = null) {
  const V = logits.length;
  const z = penalized(logits, repetitionPenalty, previous);
  const t = Math.max(temperature, 1e-8);
  for (let i = 0; i < V; i++) z[i] /= t;

  let ids;
  if (topK > 0 && topK < V) {
    // the k-th largest value without sorting the vocabulary: keep a short sorted list of the best k
    const best = new Float64Array(topK).fill(-Infinity);
    for (let i = 0; i < V; i++) {
      const v = z[i];
      if (v > best[topK - 1]) {
        let j = topK - 1;
        for (; j > 0 && best[j - 1] < v; j--) best[j] = best[j - 1];
        best[j] = v;
      }
    }
    const kth = best[topK - 1], kept = [];
    for (let i = 0; i < V; i++) if (z[i] >= kth) kept.push(i);
    ids = Int32Array.from(kept);
  } else {
    ids = new Int32Array(V);
    for (let i = 0; i < V; i++) ids[i] = i;
  }
  ids.sort((a, b) => z[b] - z[a] || a - b);   // most likely first, ties by lower id

  let n = ids.length, sum = 0;
  const probs = new Float64Array(n), max = z[ids[0]];
  for (let i = 0; i < n; i++) sum += probs[i] = Math.exp(z[ids[i]] - max);
  for (let i = 0; i < n; i++) probs[i] /= sum;
  if (topP < 1) {
    let cum = 0, keep = 0;
    while (keep < n && (keep === 0 || cum < topP)) cum += probs[keep++];   // the token that crosses topP is kept
    n = keep;
    for (let i = 0; i < n; i++) probs[i] /= cum;
  }
  return { ids: ids.subarray(0, n), probs: probs.subarray(0, n) };
}

/** Draw one token id from the result of filterLogits. */
export function sampleFrom({ ids, probs }, random = Math.random) {
  let r = random();
  for (let i = 0; i < ids.length; i++) {
    r -= probs[i];
    if (r <= 0) return ids[i];
  }
  return ids[ids.length - 1];
}

/** Choose the next token: greedy when temperature <= 0, otherwise sampled. */
export function chooseToken(logits, { temperature = 0, topK = 0, topP = 1, repetitionPenalty = 1, random = Math.random } = {}, previous = null) {
  if (temperature <= 0) {
    return repetitionPenalty === 1 || !previous || !previous.length
      ? argmax(logits)
      : argmax(penalized(logits, repetitionPenalty, previous));
  }
  return sampleFrom(filterLogits(logits, { temperature, topK, topP, repetitionPenalty }, previous), random);
}

/**
 * Choose among the tokens a constraint allows (ascending ids). Greedy takes the largest logit, ties to
 * the lower id, exactly like constrained_answers() in src/brasa/constrain.py; with a temperature the
 * other tokens are masked out and the usual sampling filters apply to what is left.
 * bias (one number per allowed token, in nats: AnswerTrie.bias) is added to the logits first.
 */
export function chooseAllowed(logits, allowed, opts = {}, previous = null, bias = null) {
  const at = bias ? (i) => logits[allowed[i]] + bias[i] : (i) => logits[allowed[i]];
  let best = allowed[0], bv = at(0);
  for (let i = 1; i < allowed.length; i++) { const v = at(i); if (v > bv) { bv = v; best = allowed[i]; } }
  const { temperature = 0, repetitionPenalty = 1 } = opts;
  if (temperature <= 0 && (repetitionPenalty === 1 || !previous || !previous.length)) return best;
  const masked = new Float32Array(logits.length).fill(-Infinity);
  for (let i = 0; i < allowed.length; i++) masked[allowed[i]] = at(i);
  const token = chooseToken(masked, opts, previous);
  return masked[token] === -Infinity ? best : token;
}

// ------------------------------------------------------------------------------------------ engine

const SCRATCH = ['kOff', 'vOff', 'x', 'h', 'qkv', 'att', 'y', 'g', 'u', 'scores', 'logitsOff'];

export class Engine {
  /**
   * @param {ArrayBuffer | Uint8Array} modelBuffer  contents of a model.*.bin file
   * @param {{ wasm?: ArrayBuffer | Uint8Array | WebAssembly.Module | null, contextLength?: number }} [opts]
   *        wasm: the bytes of kernel.wasm to use the SIMD kernel. Without it, or if it cannot be
   *        instantiated (no SIMD support), the plain JavaScript kernel is used (see kernelError).
   *        contextLength: use fewer positions than the model was trained with, to save memory.
   */
  static async create(modelBuffer, opts = {}) {
    const engine = new Engine();
    await engine._init(modelBuffer, opts);
    return engine;
  }

  async _init(modelBuffer, opts) {
    const { header, dataStart, buffer } = parseModel(modelBuffer);
    const cfg = (this.config = header.config);
    this.name = header.name || 'brasa';
    this.format = header.format;
    this.prompt = header.prompt || null;
    this.tokenizerSpec = header.tokenizer || null;
    const d = cfg.d_model, L = cfg.n_layer, H = cfg.n_head, hd = d / H, V = cfg.vocab_size, hid = cfg.mlp_hidden;
    const ctx = (this.contextLength = Math.min(opts.contextLength || cfg.block_size, cfg.block_size));
    this.d = d; this.L = L; this.H = H; this.hd = hd; this.V = V; this.hid = hid;

    // ---- plan the heap
    let top = 0;
    const alloc = (n) => { const o = top; top += n; return o; };
    const tensorOff = {};
    for (const t of header.tensors) tensorOff[t.name] = alloc(t.shape.reduce((a, b) => a * b, 1));
    this.kOff = alloc(L * H * ctx * hd);   // keys   [layer][head][pos][hd]
    this.vOff = alloc(L * H * ctx * hd);   // values [layer][head][pos][hd]
    this.x = alloc(d); this.h = alloc(d); this.qkv = alloc(3 * d); this.att = alloc(d); this.y = alloc(d);
    this.g = alloc(hid); this.u = alloc(hid); this.scores = alloc(ctx); this.logitsOff = alloc(V);
    const totalFloats = top;

    // ---- memory + kernel
    this.kernelError = null;
    if (opts.wasm) {
      try {
        const module = opts.wasm instanceof WebAssembly.Module ? opts.wasm : await WebAssembly.compile(opts.wasm);
        const reserve = 2 * 1024 * 1024; // room for the module's own data and stack below __heap_base
        const pages = Math.ceil((reserve + totalFloats * 4) / 65536);
        const memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
        const instance = await WebAssembly.instantiate(module, { env: { memory } });
        const base = instance.exports.__heap_base.value;
        if (base > reserve) throw new Error('wasm heap base beyond the reserved area');
        const baseFloats = Math.ceil(base / 16) * 4; // 16-byte aligned start, expressed in floats
        this.heap = new Float32Array(memory.buffer);
        for (const k in tensorOff) tensorOff[k] += baseFloats;
        for (const k of SCRATCH) this[k] += baseFloats;
        this.kernel = wasmKernel(instance);
      } catch (err) {
        this.kernelError = String((err && err.message) || err);   // fall back to plain JavaScript below
      }
    }
    if (!this.kernel) {
      this.heap = new Float32Array(totalFloats);
      this.kernel = jsKernel(this.heap);
    }
    this.kernelName = this.kernel.name;
    this.memoryBytes = this.heap.byteLength;

    // ---- load (and de-quantise) the weights
    const heap = this.heap;
    for (const t of header.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1), off = tensorOff[t.name], at = dataStart + t.offset;
      if (t.dtype === 'f32') {
        heap.set(new Float32Array(buffer, at, n), off);
      } else if (t.dtype === 'f16') {
        const src = new Uint16Array(buffer, at, n), table = halfTable();
        for (let i = 0; i < n; i++) heap[off + i] = table[src[i]];
      } else if (t.dtype === 'i8') {
        const src = new Int8Array(buffer, at, n);
        const rows = t.shape[0], cols = n / rows, scales = new Float32Array(buffer, dataStart + t.scalesOffset, rows);
        for (let r = 0, i = 0; r < rows; r++) {
          const s = scales[r];
          for (let c = 0; c < cols; c++, i++) heap[off + i] = src[i] * s;
        }
      } else throw new Error(`unknown tensor dtype ${t.dtype}`);
    }
    this.t = tensorOff;
    this.layers = [];
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      this.layers.push({
        norm1: tensorOff[p + 'norm1.weight'], qkv: tensorOff[p + 'attn.qkv.weight'], proj: tensorOff[p + 'attn.proj.weight'],
        norm2: tensorOff[p + 'norm2.weight'], gate: tensorOff[p + 'mlp.gate.weight'], up: tensorOff[p + 'mlp.up.weight'],
        down: tensorOff[p + 'mlp.down.weight'],
      });
    }

    // ---- rotary tables: angle(pos, i) = pos * base^(-2i/hd), computed in float32 like PyTorch
    const half = hd / 2;
    this.cos = new Float32Array(ctx * half); this.sin = new Float32Array(ctx * half);
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(cfg.rope_base, Math.fround((2 * i) / hd))));
      for (let p = 0; p < ctx; p++) {
        const a = Math.fround(p * inv);
        this.cos[p * half + i] = Math.cos(a); this.sin[p * half + i] = Math.sin(a);
      }
    }
    this.attScale = 1 / Math.sqrt(hd);
    this.eps = cfg.norm_eps;
    this.position = 0;                      // number of tokens in the cache
    this.tokens = new Int32Array(ctx);      // the tokens in the cache (for prefix reuse)
    this.reusedTokens = 0;
    this.logits = heap.subarray(this.logitsOff, this.logitsOff + V);
    this._attn = null; this._norms = null; this._captured = false;
  }

  /** Forget the context (start a new sequence). The cache is simply overwritten, nothing to free. */
  reset() { this.position = 0; }

  /** Drop everything after the first `position` tokens (keeps the cache of the prefix). */
  rewind(position) {
    if (position < 0 || position > this.position) throw new RangeError(`cannot rewind to ${position}`);
    this.position = position;
  }

  /**
   * Turn inspection buffers on or off. With attention: true every later step() records the attention
   * weights of all layers and heads (read them with attention()); with norms: true it records the L2 norm
   * of the residual stream after the embedding and after every block (residualNorms()).
   */
  setCapture({ attention = false, norms = false } = {}) {
    if (attention && !this._attn) this._attn = new Float32Array(this.L * this.H * this.contextLength);
    if (!attention) this._attn = null;
    if (norms && !this._norms) this._norms = new Float32Array(this.L + 1);
    if (!norms) this._norms = null;
    this._captured = false;
  }

  /**
   * Attention of the last step: for every layer and head, the weights its position gave to each position
   * of the context (rows sum to 1). data[(layer * heads + head) * length + position].
   * Null until a step has run with attention capture on.
   */
  attention() {
    if (!this._attn || !this._captured) return null;
    const { L, H, contextLength: ctx } = this, n = this.position;
    const data = new Float32Array(L * H * n);
    for (let i = 0; i < L * H; i++) data.set(this._attn.subarray(i * ctx, i * ctx + n), i * n);
    return { layers: L, heads: H, length: n, data };
  }

  /** Residual-stream norms of the last step: [after embedding, after block 0, ..., after block L-1]. */
  residualNorms() {
    return this._norms && this._captured ? Float32Array.from(this._norms) : null;
  }

  _rmsnorm(src, dst, gain) {
    const heap = this.heap, d = this.d;
    let ss = 0;
    for (let i = 0; i < d; i++) { const v = heap[src + i]; ss += v * v; }
    const r = 1 / Math.sqrt(ss / d + this.eps);
    for (let i = 0; i < d; i++) heap[dst + i] = heap[src + i] * r * heap[gain + i];
  }

  _l2(src) {
    const heap = this.heap, d = this.d;
    let ss = 0;
    for (let i = 0; i < d; i++) { const v = heap[src + i]; ss += v * v; }
    return Math.sqrt(ss);
  }

  /** Feed one token at the next position. Returns the logits view (valid until the next call) or null. */
  step(token, wantLogits = true) {
    const pos = this.position;
    if (pos >= this.contextLength) throw new Error('context is full');
    if (!(token >= 0 && token < this.V)) throw new RangeError(`token id ${token} is outside the vocabulary`);
    const { heap, kernel, d, H, hd, hid, contextLength: ctx } = this;
    const half = hd >> 1, rp = pos * half, cos = this.cos, sin = this.sin;
    const { x, h, qkv, att, y, g, u, scores } = this;
    const attn = this._attn, norms = this._norms;
    const emb = this.t['tok_emb.weight'];
    heap.copyWithin(x, emb + token * d, emb + (token + 1) * d);
    if (norms) norms[0] = this._l2(x);

    for (let l = 0; l < this.L; l++) {
      const w = this.layers[l];
      // ---- attention
      this._rmsnorm(x, h, w.norm1);
      kernel.matvec(w.qkv, h, qkv, 3 * d, d);
      for (let hh = 0; hh < H; hh++) {
        const qb = qkv + hh * hd, kb = qkv + d + hh * hd, vb = qkv + 2 * d + hh * hd;
        const kBase = this.kOff + (l * H + hh) * ctx * hd, vBase = this.vOff + (l * H + hh) * ctx * hd;
        const kc = kBase + pos * hd;
        for (let i = 0; i < half; i++) {   // rotate q in place, write the rotated k straight into the cache
          const c = cos[rp + i], s = sin[rp + i];
          const q1 = heap[qb + i], q2 = heap[qb + half + i];
          heap[qb + i] = q1 * c - q2 * s; heap[qb + half + i] = q1 * s + q2 * c;
          const k1 = heap[kb + i], k2 = heap[kb + half + i];
          heap[kc + i] = k1 * c - k2 * s; heap[kc + half + i] = k1 * s + k2 * c;
        }
        heap.copyWithin(vBase + pos * hd, vb, vb + hd);
        const n = pos + 1;
        kernel.matvec(kBase, qb, scores, n, hd);          // scores[t] = k_t . q
        let max = -Infinity;
        for (let t = 0; t < n; t++) { const v = heap[scores + t] * this.attScale; heap[scores + t] = v; if (v > max) max = v; }
        let sum = 0;
        for (let t = 0; t < n; t++) { const e = Math.exp(heap[scores + t] - max); heap[scores + t] = e; sum += e; }
        const inv = 1 / sum;
        for (let t = 0; t < n; t++) heap[scores + t] *= inv;
        if (attn) attn.set(heap.subarray(scores, scores + n), (l * H + hh) * ctx);
        kernel.tmatvec(vBase, scores, att + hh * hd, n, hd); // weighted sum of the cached values
      }
      kernel.matvec(w.proj, att, y, d, d);
      for (let i = 0; i < d; i++) heap[x + i] += heap[y + i];

      // ---- feed-forward
      this._rmsnorm(x, h, w.norm2);
      if (w.gate !== undefined) {          // SwiGLU: down( silu(gate h) * (up h) )
        kernel.matvec(w.gate, h, g, hid, d);
        kernel.matvec(w.up, h, u, hid, d);
        for (let i = 0; i < hid; i++) { const v = heap[g + i]; heap[g + i] = (v / (1 + Math.exp(-v))) * heap[u + i]; }
        kernel.matvec(w.down, g, y, d, hid);
      } else {                             // GELU (tanh approximation)
        kernel.matvec(w.up, h, u, hid, d);
        for (let i = 0; i < hid; i++) {
          const v = heap[u + i];
          heap[u + i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        }
        kernel.matvec(w.down, u, y, d, hid);
      }
      for (let i = 0; i < d; i++) heap[x + i] += heap[y + i];
      if (norms) norms[l + 1] = this._l2(x);
    }
    this.tokens[pos] = token;
    this.position = pos + 1;
    this._captured = true;
    if (!wantLogits) return null;
    this._rmsnorm(x, h, this.t['norm_f.weight']);
    kernel.matvec(emb, h, this.logitsOff, this.V, d);   // tied output head
    return this.logits;
  }

  /** Feed a sequence after the current position; returns the logits after its last token. */
  prefill(tokens) {
    let out = null;
    for (let i = 0; i < tokens.length; i++) out = this.step(tokens[i], i === tokens.length - 1);
    return out;
  }

  /**
   * Make the cache hold exactly `tokens` and return the logits after the last one. Positions that
   * already hold the same tokens are kept, so repeating a prompt, or extending the previous sequence by
   * a few tokens, only pays for what changed (reusedTokens says how many were kept).
   */
  feed(tokens) {
    if (!tokens.length) throw new Error('empty prompt');
    if (tokens.length > this.contextLength) {
      throw new Error(`prompt of ${tokens.length} tokens exceeds the context length ${this.contextLength}`);
    }
    let common = 0;
    const max = Math.min(this.position, tokens.length);
    while (common < max && this.tokens[common] === tokens[common]) common++;
    if (common === tokens.length) common--;   // the last token is always re-run to get its logits
    this.position = this.reusedTokens = common;
    let out = null;
    for (let i = common; i < tokens.length; i++) out = this.step(tokens[i], i === tokens.length - 1);
    return out;
  }

  /**
   * The model read as an encoder: one unit vector for a token sequence, pooled from the final hidden states
   * (after the last RMSNorm, before the output head) exactly as src/brasa/encoder.py does. 'mean' averages
   * every position, 'last' takes the final one. The dense half of the retrieval index compares this vector
   * of a question with the precomputed vectors of the facts. The context is overwritten: the next feed()
   * starts from an empty cache.
   * @param {ArrayLike<number>} tokens
   * @param {'mean'|'last'} [pooling='mean']
   * @returns {Float32Array} d_model values, L2 norm 1
   */
  embed(tokens, pooling = 'mean') {
    if (!tokens.length) throw new Error('nothing to embed');
    if (tokens.length > this.contextLength) {
      throw new Error(`sequence of ${tokens.length} tokens exceeds the context length ${this.contextLength}`);
    }
    if (pooling !== 'mean' && pooling !== 'last') throw new Error(`unknown pooling: ${pooling}`);
    const { heap, d, x, h } = this, gain = this.t['norm_f.weight'];
    const sum = new Float64Array(d);
    this.position = 0;
    for (let i = 0; i < tokens.length; i++) {
      this.step(tokens[i], false);
      if (pooling === 'mean' || i === tokens.length - 1) {
        this._rmsnorm(x, h, gain);
        for (let j = 0; j < d; j++) sum[j] += heap[h + j];
      }
    }
    this.position = this.reusedTokens = 0;   // the cache holds the embedded text, not a prompt: start over
    this._captured = false;
    let ss = 0;
    for (let j = 0; j < d; j++) ss += sum[j] * sum[j];
    const inv = 1 / Math.max(Math.sqrt(ss), 1e-12);
    const out = new Float32Array(d);
    for (let j = 0; j < d; j++) out[j] = sum[j] * inv;
    return out;
  }

  /**
   * Generate token by token. Yields { index, token, stop } for every new token, plus what `inspect` asks:
   *   inspect.top: k      -> top: [{ id, prob, logit }]  the model's k most likely tokens at this step
   *                          prob: probability the model gave to the chosen token
   *   inspect.attention   -> attention: see attention()  (of the position that predicted this token)
   *   inspect.norms       -> norms: see residualNorms()
   * The generator returns 'stop' (a stop token was produced) or 'length' (the budget or the context ran out).
   * The repetition penalty looks at the generated tokens only, never at the prompt.
   *
   * opts.constraint (an AnswerTrie of retrieval.js, or anything with eos, start(), allowed(state) and
   * advance(state, token)) restricts every step to the tokens it allows and stops at its eos. Each step
   * then carries `choices`, the number of allowed tokens: 1 means the token was forced, and a forced
   * token is fed without computing the output head unless `inspect` needs the distribution. A constraint
   * with `hasPrior` and bias(state) (the retriever's ranking as a prior) has that bias added to the logits
   * of the allowed tokens before the choice.
   */
  *stream(promptTokens, opts = {}) {
    const { maxNewTokens = 128, stopTokens = [], inspect = null, constraint = null } = opts;
    const stops = new Set(stopTokens);
    if (constraint) stops.add(constraint.eos);
    this.setCapture({ attention: !!(inspect && inspect.attention), norms: !!(inspect && inspect.norms) });
    let logits = this.feed(promptTokens);
    const budget = Math.min(maxNewTokens, this.contextLength - promptTokens.length);
    const generated = [];
    let state = constraint ? constraint.start() : null;
    for (let i = 0; i < budget; i++) {
      let token, allowed = null;
      if (constraint) {
        allowed = constraint.allowed(state);
        if (!allowed.length) return 'stop';
        token = allowed.length === 1 ? allowed[0]
          : chooseAllowed(logits, allowed, opts, generated, constraint.hasPrior ? constraint.bias(state) : null);
      } else {
        token = chooseToken(logits, opts, generated);
      }
      const step = { index: i, token, stop: stops.has(token) };
      if (allowed) step.choices = allowed.length;
      if (inspect) {
        if (inspect.top) step.top = topTokens(logits, inspect.top);
        step.prob = probabilityOf(logits, token);
        if (inspect.attention) step.attention = this.attention();
        if (inspect.norms) step.norms = this.residualNorms();
      }
      generated.push(token);
      yield step;
      if (step.stop) return 'stop';
      if (i + 1 >= budget) break;
      if (constraint) {
        state = constraint.advance(state, token);
        logits = this.step(token, !!inspect || constraint.allowed(state).length !== 1);
      } else {
        logits = this.step(token, true);
      }
    }
    return 'length';
  }

  /**
   * Run stream() to the end. onToken(step) is called for every token; return false from it to stop.
   * @returns {{ tokens: number[], stopReason: 'stop'|'length'|'aborted', promptTokens: number,
   *             reusedTokens: number, prefillMs: number, decodeMs: number, tokensPerSecond: number }}
   */
  generate(promptTokens, opts = {}) {
    const t0 = now();
    const tokens = [];
    const it = this.stream(promptTokens, opts);
    let t1 = null, stopReason = 'length';
    for (;;) {
      const r = it.next();
      if (t1 === null) t1 = now();   // prompt processed and first token chosen
      if (r.done) { stopReason = r.value; break; }
      tokens.push(r.value.token);
      if (opts.onToken && opts.onToken(r.value) === false) { stopReason = 'aborted'; break; }
    }
    const decodeMs = now() - t1;
    return {
      tokens, stopReason, promptTokens: promptTokens.length, reusedTokens: this.reusedTokens,
      prefillMs: t1 - t0, decodeMs,
      tokensPerSecond: tokens.length > 1 && decodeMs > 0 ? ((tokens.length - 1) * 1000) / decodeMs : 0,
    };
  }
}
