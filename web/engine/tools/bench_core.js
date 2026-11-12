// Shared benchmark routine: used by the Node script and usable from a page or a Web Worker.
// cfg = { model: ArrayBuffer, wasm?: ArrayBuffer, prompt: number[], maxNew: number, repeats?: number }
import { Engine, argmax } from '../engine.js';

export async function benchEngine(cfg, now = () => performance.now()) {
  const t0 = now();
  const engine = await Engine.create(cfg.model, cfg.wasm ? { wasm: cfg.wasm } : {});
  const initMs = now() - t0;
  const budget = Math.min(cfg.maxNew, engine.contextLength - cfg.prompt.length);
  const runs = [];
  for (let r = 0; r < (cfg.repeats || 3); r++) {
    engine.reset();
    const stepMs = new Float64Array(budget);
    const ts = now();
    let logits = engine.prefill(cfg.prompt);
    let id = argmax(logits);
    const ttftMs = now() - ts;                 // prompt processed and first token chosen
    let last = now();
    for (let i = 1; i < budget; i++) {
      logits = engine.step(id, true);
      id = argmax(logits);
      const t = now();
      stepMs[i] = t - last;
      last = t;
    }
    const decodeMs = last - ts - ttftMs;
    const window = (a, b) => {                 // tokens/s over the generated tokens with index in [a, b)
      let s = 0, n = 0;
      for (let i = Math.max(1, a); i < Math.min(b, budget); i++) { s += stepMs[i]; n++; }
      return n ? Math.round((1000 * n) / s * 10) / 10 : null;
    };
    runs.push({
      ttftMs: Math.round(ttftMs * 100) / 100,
      prefillTokPerSec: Math.round((1000 * cfg.prompt.length) / ttftMs),
      decodeTokPerSec: Math.round(((budget - 1) * 1000) / decodeMs * 10) / 10,
      first32: window(1, 32), last32: window(budget - 32, budget),
    });
  }
  return {
    kernel: engine.kernelName, config: engine.config, memoryMB: Math.round(engine.memoryBytes / 1048576 * 10) / 10,
    initMs: Math.round(initMs * 10) / 10, promptTokens: cfg.prompt.length, newTokens: budget, runs,
  };
}
