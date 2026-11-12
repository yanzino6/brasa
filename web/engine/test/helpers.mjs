// Shared helpers for the Node tests: everything is read from tests/fixtures/tiny, a tiny random model
// exported by scripts/make_fixtures.py.
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

export const FIXTURES = fileURLToPath(new URL('../../../tests/fixtures/tiny/', import.meta.url));
export const WASM_PATH = fileURLToPath(new URL('../../wasm/kernel.wasm', import.meta.url));

export const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
export const readBuffer = (name) => toArrayBuffer(fs.readFileSync(FIXTURES + name));
export const readJson = (name) => JSON.parse(fs.readFileSync(FIXTURES + name, 'utf8'));
export const wasmBytes = () => toArrayBuffer(fs.readFileSync(WASM_PATH));
