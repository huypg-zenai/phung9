// gemma_quant.cuh - fused Gemma add+RMSNorm + dynamic per-token FP8 E4M3 quant (kernel-optim fusedq step 2, 21/10).
// Byte-identical by construction to sgl_kernel.gemma_fused_add_rmsnorm (flashinfer @bc29697b norm.cuh
// FusedAddRMSNormKernel, weight_bias 1, built --use_fast_math in the SM90 library) followed by SGLang's JIT
// per_token_quant_fp8_cta_kernel: the norm lines (thread layout, sum-of-squares reduction order, rsqrt.approx, fp32 x
// kept in smem, (bias + w) step) and the quant lines are copied; build with --use_fast_math on sm90.
// Writes the same in-place results as the original (input <- normed bf16, residual <- input + residual) PLUS
// output_q / output_s = per-token fp8 quant of the normed row. One token per CTA; valid while the original quant would
// use its CTA variant (num_tokens < 2 * SMs * 8).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>

namespace sglang {

namespace gq {
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {  // flashinfer math.cuh
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(y) : "f"(x), "r"(lane_mask));
  return y;
}
__forceinline__ __device__ float rsqrt(float x) {  // flashinfer math.cuh
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__host__ __device__ __forceinline__ uint32_t ceil_div(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}
}  // namespace gq

template <uint32_t VEC_SIZE, typename T>
__global__ void gemma_add_rmsnorm_quant_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    float weight_bias,
    float eps) {
  using namespace device;
  using tvec_t = AlignedVector<T, VEC_SIZE>;
  using qvec_t = AlignedVector<fp8_e4m3_t, VEC_SIZE>;
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = gq::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ __align__(16) float smem[];
  float* smem_x = smem + gq::ceil_div(num_warps, 4) * 4;
  T* smem_y = reinterpret_cast<T*>(smem_x + d);  // normed bf16 row (for the quant pass)

  // ---- flashinfer FusedAddRMSNormKernel, pass 1 (verbatim arithmetic)
  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    const uint32_t e = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    tvec_t input_vec, residual_vec;
    float x_vec[VEC_SIZE];  // AlignedVector<float, 8> would exceed 16 B; values are the same
    input_vec.fill(T(0.f));
    residual_vec.fill(T(0.f));
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) x_vec[j] = 0.f;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + e);
      residual_vec.load(residual + bx * stride_residual + e);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + e);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) smem_x[e + j] = x_vec[j];
    }
  }
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += gq::shfl_xor_sync(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += gq::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();
  float rms_rcp = gq::rsqrt(smem[0] / float(d) + eps);

  // ---- pass 2 (verbatim) + running amax of the bf16 result
  float max_value = 0.0f;
  for (uint32_t i = 0; i < rounds; i++) {
    const uint32_t e = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    tvec_t input_vec, weight_vec;
    float x_vec[VEC_SIZE];
    input_vec.fill(T(0.f));
    weight_vec.fill(T(0.f));
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) x_vec[j] = 0.f;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + e);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) x_vec[j] = smem_x[e + j];
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + e);
      input_vec.store(smem_y + e);
    }
  }

  // ---- per_token_quant_fp8_cta_kernel arithmetic (max is exact: any reduction order gives the same amax)
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    max_value = math::max(max_value, gq::shfl_xor_sync(max_value, offset));
  }
  __shared__ float red_max[32];
  __shared__ float scale_smem;
  __syncthreads();  // smem[0] (sum) no longer read; smem_y complete after the next barrier
  if (tx == 0) red_max[ty] = max_value;
  __syncthreads();
  if (thread_id == 0) {
    float m = 0.0f;
    for (uint32_t w = 0; w < num_warps; ++w) m = math::max(m, red_max[w]);
    scale_smem = m / math::FP8_E4M3_MAX;
    output_s[bx] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;
  for (uint32_t i = 0; i < rounds; i++) {
    const uint32_t e = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      tvec_t input_vec;
      qvec_t output_vec;
      input_vec.load(smem_y + e);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const float value = static_cast<float>(input_vec[j]) * scale_inv;
        output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
      }
      output_vec.store(output_q + static_cast<size_t>(bx) * d + e);
    }
  }
}

/** input/residual [M, d] (in place, like gemma_fused_add_rmsnorm), weight [d], output_q [>=M, d] fp8, output_s [>=M,1]. */
template <typename T>
void gemma_add_rmsnorm_quant_fp8(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output_q,
    tvm::ffi::TensorView output_s,
    double eps) {
  using namespace host;
  auto M = SymbolicSize{"num_tokens"};
  auto MOutput = SymbolicSize{"output_num_tokens"};
  auto D = SymbolicSize{"hidden"};
  auto SI = SymbolicSize{"stride_input"};
  auto SR = SymbolicSize{"stride_residual"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, D}).with_strides({SI, 1}).with_dtype<T>().with_device(device).verify(input);
  TensorMatcher({M, D}).with_strides({SR, 1}).with_dtype<T>().with_device(device).verify(residual);
  TensorMatcher({D}).with_dtype<T>().with_device(device).verify(weight);
  TensorMatcher({MOutput, D}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
  TensorMatcher({MOutput, 1}).with_dtype<float>().with_device(device).verify(output_s);
  const uint32_t batch_size = static_cast<uint32_t>(M.unwrap());
  const uint32_t d = static_cast<uint32_t>(D.unwrap());
  CHECK_HOST(MOutput.unwrap() >= M.unwrap()) << "gemma_add_rmsnorm_quant_fp8: output rows";
  constexpr uint32_t vec_size = 16 / sizeof(T);  // flashinfer: gcd(16 / sizeof(T), d)
  CHECK_HOST(d % vec_size == 0) << "gemma_add_rmsnorm_quant_fp8: d must be a multiple of " << vec_size;
  if (batch_size == 0) return;
  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = gq::ceil_div(block_size, 32);
  const size_t smem_size = (gq::ceil_div(num_warps, 4) * 4 + d) * sizeof(float) + d * sizeof(T);
  auto kernel = gemma_add_rmsnorm_quant_kernel<vec_size, T>;
  const cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
  CHECK_HOST(err == cudaSuccess) << "gemma_add_rmsnorm_quant_fp8: cudaFuncSetAttribute failed";
  LaunchKernel(dim3(batch_size), dim3(32, num_warps), device.unwrap(), smem_size)(
      kernel,
      static_cast<T*>(input.data_ptr()),
      static_cast<T*>(residual.data_ptr()),
      static_cast<const T*>(weight.data_ptr()),
      static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      d,
      static_cast<uint32_t>(SI.unwrap()),
      static_cast<uint32_t>(SR.unwrap()),
      1.0f,
      static_cast<float>(eps));
}

}  // namespace sglang
