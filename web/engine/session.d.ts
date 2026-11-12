import type { WorkerRequest, WorkerResponse } from './worker.js';

/** The worker protocol without the worker: handle() takes requests, `post` receives the responses. */
export class Session {
  constructor(
    post: (message: WorkerResponse, transfer?: Transferable[]) => void,
    deps?: { fetch?: typeof fetch; sliceMs?: number },
  );
  handle(message: WorkerRequest): Promise<void>;
}
