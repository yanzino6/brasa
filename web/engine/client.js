// Main-thread side of the worker protocol: turns the message stream into promises and callbacks.
//
//   const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
//   const brasa = new BrasaClient(worker);
//   await brasa.load({ model: '/brasa/model.i8.bin', wasm: '/brasa/kernel.wasm', retrieval: '/brasa/retrieval.json' });
//   const run = brasa.ask('Onde ele mora?', { inspect: { top: 5 } }, { onToken: (t) => render(t.delta) });
//   const { text, stats } = await run.done;       // run.abort() stops it early
//
// `worker` is anything with postMessage() and addEventListener('message') or onmessage: a Worker, or in
// tests an in-process stand-in.

export class BrasaClient {
  constructor(worker) {
    this.worker = worker;
    this.nextId = 1;
    this.pending = new Map();   // request id -> { resolve, reject, handlers }
    const onMessage = (event) => this._receive(event.data);
    if (worker.addEventListener) worker.addEventListener('message', onMessage);
    else worker.onmessage = onMessage;
  }

  _receive(message) {
    const entry = this.pending.get(message.id);
    if (!entry) return;
    const { handlers } = entry;
    switch (message.type) {
      case 'progress': if (handlers.onProgress) handlers.onProgress(message); return;
      case 'context': if (handlers.onContext) handlers.onContext(message); return;
      case 'prompt': if (handlers.onPrompt) handlers.onPrompt(message); return;
      case 'token': if (handlers.onToken) handlers.onToken(message); return;
      case 'error':
        this.pending.delete(message.id);
        entry.reject(new Error(message.message));
        return;
      default:   // ready, tokens, hits, done, unloaded: the final answer of a request
        this.pending.delete(message.id);
        entry.resolve(message);
    }
  }

  _request(message, handlers = {}, transfer = []) {
    const id = this.nextId++;
    const done = new Promise((resolve, reject) => this.pending.set(id, { resolve, reject, handlers }));
    this.worker.postMessage({ ...message, id }, transfer);
    return { id, done };
  }

  /** Fetch and initialise the model. Resolves with the model information of the "ready" message. */
  load(sources, onProgress) {
    return this._request({ type: 'load', ...sources }, { onProgress }).done.then((m) => m.info);
  }

  /** Token ids and boundaries of a text. */
  tokenize(text, { allowSpecial = false } = {}) {
    return this._request({ type: 'tokenize', text, allowSpecial }).done.then(({ text: normalized, tokens }) => ({ text: normalized, tokens }));
  }

  /** The k best facts for a query. */
  retrieve(query, k, lang) {
    return this._request({ type: 'retrieve', query, k, lang }).done.then((m) => m.hits);
  }

  /** Ask a question. Returns { id, done, abort }; done resolves with the "done" message. */
  ask(question, options = {}, handlers = {}) {
    const run = this._request({ type: 'ask', question, ...options }, handlers);
    return { ...run, abort: () => this.abort(run.id) };
  }

  /** Continue an arbitrary token sequence. Returns { id, done, abort }. */
  generate(tokens, options = {}, handlers = {}) {
    const run = this._request({ type: 'generate', tokens, ...options }, handlers);
    return { ...run, abort: () => this.abort(run.id) };
  }

  /** Stop a generation (default: whichever is running). Its promise still resolves, with stopReason "aborted". */
  abort(id) {
    this.worker.postMessage({ type: 'abort', id });
  }

  /** Free the model inside the worker. */
  unload() {
    return this._request({ type: 'unload' }).done.then(() => undefined);
  }

  /** Terminate the worker and reject everything still pending. */
  dispose() {
    if (this.worker.terminate) this.worker.terminate();
    for (const entry of this.pending.values()) entry.reject(new Error('the worker was terminated'));
    this.pending.clear();
  }
}
