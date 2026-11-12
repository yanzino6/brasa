// Two tiny linear-algebra kernels for the browser inference engine, compiled to WebAssembly with SIMD.
// No libc, no allocation, no globals: JavaScript owns the memory and passes byte offsets.
//
//   matvec : y[r]  = sum_c W[r, c] * x[c]          (dense layer applied to one vector)
//   tmatvec: out[c] = sum_r a[r] * V[r, c]         (attention: weighted sum of cached value rows)
//
// build: see build.sh  (zig cc --target=wasm32-freestanding -O3 -msimd128)
#include <wasm_simd128.h>

#define EXPORT(name) __attribute__((export_name(name)))

static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) +
         wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

EXPORT("matvec")
void matvec(const float *W, const float *x, float *y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const float *w = W + (unsigned)r * (unsigned)cols;
    v128_t a0 = wasm_f32x4_splat(0.0f), a1 = a0, a2 = a0, a3 = a0;
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(w + c), wasm_v128_load(x + c)));
      a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load(w + c + 4), wasm_v128_load(x + c + 4)));
      a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wasm_v128_load(w + c + 8), wasm_v128_load(x + c + 8)));
      a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wasm_v128_load(w + c + 12), wasm_v128_load(x + c + 12)));
    }
    for (; c + 4 <= cols; c += 4)
      a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(w + c), wasm_v128_load(x + c)));
    float s = hsum(wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3)));
    for (; c < cols; c++) s += w[c] * x[c];
    y[r] = s;
  }
}

EXPORT("tmatvec")
void tmatvec(const float *V, const float *a, float *out, int rows, int cols) {
  for (int c = 0; c < cols; c++) out[c] = 0.0f;
  for (int r = 0; r < rows; r++) {
    const float *v = V + (unsigned)r * (unsigned)cols;
    const float ar = a[r];
    const v128_t s = wasm_f32x4_splat(ar);
    int c = 0;
    for (; c + 4 <= cols; c += 4)
      wasm_v128_store(out + c, wasm_f32x4_add(wasm_v128_load(out + c), wasm_f32x4_mul(s, wasm_v128_load(v + c))));
    for (; c < cols; c++) out[c] += ar * v[c];
  }
}
