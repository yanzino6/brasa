import type {
  AskRequest, ContextMessage, DoneMessage, GenerateRequest, LoadRequest, ModelInfo, ProgressMessage,
  PromptMessage, TokenMessage,
} from './worker.js';
import type { Hit } from './retrieval.js';
import type { TokenSpan } from './tokenizer.js';

/** The part of Worker the client uses: a real Worker fits, and so does an in-process stand-in. */
export interface WorkerLike {
  postMessage(message: any, transfer?: any): void;
  addEventListener?: (type: 'message', listener: (event: any) => void) => void;
  onmessage?: any;
  terminate?: () => void;
}

export interface GenerationHandlers {
  onContext?: (message: ContextMessage) => void;
  onPrompt?: (message: PromptMessage) => void;
  onToken?: (message: TokenMessage) => void;
}

export interface Run {
  id: number;
  /** Resolves with the final message; rejects on an error message. */
  done: Promise<DoneMessage>;
  /** Stop early; `done` still resolves, with stats.stopReason === 'aborted'. */
  abort(): void;
}

export class BrasaClient {
  constructor(worker: WorkerLike);
  load(sources: Omit<LoadRequest, 'type' | 'id'>, onProgress?: (message: ProgressMessage) => void): Promise<ModelInfo>;
  tokenize(text: string, options?: { allowSpecial?: boolean }): Promise<{ text: string; tokens: TokenSpan[] }>;
  retrieve(query: string, k?: number, lang?: 'pt' | 'en' | null): Promise<Hit[]>;
  ask(question: string, options?: Omit<AskRequest, 'type' | 'id' | 'question'>, handlers?: GenerationHandlers): Run;
  generate(tokens: number[], options?: Omit<GenerateRequest, 'type' | 'id' | 'tokens'>, handlers?: GenerationHandlers): Run;
  abort(id?: number): void;
  unload(): Promise<void>;
  dispose(): void;
}
