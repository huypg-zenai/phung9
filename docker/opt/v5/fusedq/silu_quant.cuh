// silu_quant.cuh - fused SiLU-and-mul + dynamic per-token FP8 E4M3 quant (kernel-optim fusedq, 21/10).
// Byte-identical by construction to SGLang @1cf2b8c54 JIT act_and_mul_kernel<T, kSiLU> (non-rounding) followed by
// per_token_quant_fp8_cta_kernel: the activation lines are copied from jit/csrc/elementwise/activation.cuh and the
// quant lines from jit/csrc/gemm/per_token_quant_fp8.cuh; build with --use_fast_math on sm90 like both originals.
// One token per CTA (the original quant's CTA variant, used for num_tokens < 2 * SMs * 8); the bf16 activation row
// stays in shared memory between the amax pass and the quant pass. Larger batches must use the unfused path.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace sglang {

constexpr uint32_t kFusedQuantWarpSize = 32;

SGL_DEVICE float fq_silu_f32(float x_f32) {
  return x_f32 / (1.0f + expf(-x_f32));  // == apply_activation_f32<kSiLU>
}

template <typename T, int kVecSize>
__global__ void silu_mul_quant_fp8_cta_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;
  extern __shared__ __align__(16) uint8_t fq_smem_raw[];
  T* act_row = reinterpret_cast<T*>(fq_smem_raw);

  const uint32_t token_id = blockIdx.x;
  const uint32_t num_vecs = hidden_dim / kVecSize;
  const T* token_input = input + static_cast<size_t>(token_id) * hidden_dim * 2;
  fp8_e4m3_t* token_output = output_q + static_cast<size_t>(token_id) * hidden_dim;

  float max_value = 0.0f;
  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    vec_t gate, up, out;
    gate.load(token_input, i);
    up.load(token_input + hidden_dim, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float gate_f32 = device::cast<fp32_t>(gate[j]);
      const float up_f32 = device::cast<fp32_t>(up[j]);
      out[j] = device::cast<T>(fq_silu_f32(gate_f32) * up_f32);
      max_value = math::max(max_value, math::abs(static_cast<float>(out[j])));
    }
    out.store(act_row, i);
  }

  __shared__ float reduction_smem[kFusedQuantWarpSize];
  __shared__ float scale_smem;
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    scale_smem = reduction_smem[0] / math::FP8_E4M3_MAX;
    output_s[token_id] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;

  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(act_row, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(token_output, i);
  }
}

/** input [M, 2H] (gate | up), output_q [>=M, H] fp8, output_s [>=M, 1] fp32. */
template <typename T>
void silu_mul_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;
  auto M = SymbolicSize{"num_tokens"};
  auto MOutput = SymbolicSize{"output_num_tokens"};
  auto K2 = SymbolicSize{"two_hidden"};
  auto K = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({M, K2}).with_dtype<T>().with_device(device).verify(input);
  TensorMatcher({MOutput, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
  TensorMatcher({MOutput, 1}).with_dtype<float>().with_device(device).verify(output_s);
  constexpr uint32_t kVec = 16 / sizeof(T);
  CHECK_HOST(K2.unwrap() == 2 * K.unwrap()) << "silu_mul_quant_fp8: input must be [M, 2H]";
  CHECK_HOST(MOutput.unwrap() >= M.unwrap()) << "silu_mul_quant_fp8: output rows";
  CHECK_HOST(K.unwrap() % kVec == 0) << "silu_mul_quant_fp8: H must be a multiple of the vector size";
  CHECK_HOST(K.unwrap() * sizeof(T) <= 96 * 1024) << "silu_mul_quant_fp8: row too large for shared memory";
  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden_dim = static_cast<uint32_t>(K.unwrap());
  if (num_tokens == 0) return;
  const size_t smem = static_cast<size_t>(hidden_dim) * sizeof(T);
  auto kernel = silu_mul_quant_fp8_cta_kernel<T, kVec>;
  if (smem > 48 * 1024) {
    const cudaError_t err =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    CHECK_HOST(err == cudaSuccess) << "silu_mul_quant_fp8: cudaFuncSetAttribute failed";
  }
  LaunchKernel(num_tokens, 256, device.unwrap(), smem)(
      kernel,
      static_cast<const T*>(input.data_ptr()),
      static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      hidden_dim);
}

}  // namespace sglang
