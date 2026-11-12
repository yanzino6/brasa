// Web Worker entry point: runs the engine off the main thread so that the page never freezes.
//
//   const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
//   worker.postMessage({ type: 'load', id: 1, model: '/brasa/model.i8.bin', wasm: '/brasa/kernel.wasm',
//                        retrieval: '/brasa/retrieval.json' });
//
// The protocol is documented in worker.d.ts and implemented in session.js; client.js wraps it in promises.

import { Session } from './session.js';

const session = new Session((message, transfer) => self.postMessage(message, transfer || []));
self.onmessage = (event) => session.handle(event.data);
