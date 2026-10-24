// kopt_wq_gemm.cuh - served copy of the WAVEQ kernel (kernel-optim gemm/waveq_kernel.cu): production sgl-kernel
// 0.4.6.post1 SM90 FP8 rowwise GEMM (struct + caller verbatim from fp8_gemm_sm90_dispatch.cuh:12-120 / :151-183,
// epilogue c3x::ScaledEpilogue: D = bf16_rn(sa[m] * (sb[n] * acc))), tile / cluster / schedule as template params.
// Included by the per-config translation units that apply_wq.py --build generates.
#pragma once
// include block = production fp8_gemm_kernel.cu:20-48 (order matters for the cutlass EVT headers)
#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <torch/extension.h>

// = the includes of fp8_gemm_sm90_dispatch.cuh:6-8
#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "cutlass_extensions/gemm/cutlass_gemm_caller.cuh"

using namespace cute;

// ---------------- verbatim from fp8_gemm_sm90_dispatch.cuh:12-120 ----------------
template <
    typename ElementAB_,
    typename ElementD_,
    template <typename, typename, typename> typename Epilogue_,
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    bool swap_ab_ = false>
struct cutlass_3x_gemm_sm90_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc = typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t, float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr bool swap_ab = swap_ab_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc,
      float,
      ElementC,
      conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
      AlignmentCD,
      ElementD,
      conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
      AlignmentCD,
      EpilogueSchedule,
      EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize = sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutB_T,
          AlignmentAB,
          ElementAB,
          LayoutA_T,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutA,
          AlignmentAB,
          ElementAB,
          LayoutB,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp>;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

// ---------------- verbatim from fp8_gemm_sm90_dispatch.cuh:151-183 ----------------
template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm90_fp8(
    torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, EpilogueArgs&&... epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr, a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...), c_ptr, c_stride, c_ptr, c_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args, epilogue_args);
}


// ---------------- template-parameterised entry (one explicit use per generated config TU) ----------------
template <int S>
struct KoptWqSched;
template <>
struct KoptWqSched<0> {  // pingpong (production M>128 schedule)
  using K = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using E = cutlass::epilogue::TmaWarpSpecialized;
};
template <>
struct KoptWqSched<1> {  // cooperative
  using K = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using E = cutlass::epilogue::TmaWarpSpecializedCooperative;
};
template <>
struct KoptWqSched<2> {  // plain warp-specialized (production M<=64 schedule)
  using K = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using E = cutlass::epilogue::TmaWarpSpecialized;
};

template <int S, int TM, int TN, int TK, int CM, int CN>
torch::Tensor kopt_wq_mm(
    torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& scales_a, torch::Tensor const& scales_b) {
  using Gemm = cutlass_3x_gemm_sm90_fp8<
      cutlass::float_e4m3_t,
      cutlass::bfloat16_t,
      c3x::ScaledEpilogue,
      Shape<Int<TM>, Int<TN>, Int<TK>>,
      Shape<Int<CM>, Int<CN>, _1>,
      typename KoptWqSched<S>::K,
      typename KoptWqSched<S>::E>;
  // same input contract as sgl_kernel.fp8_scaled_mm (fp8_gemm_kernel.cu:1125-1180), bf16 out, no bias
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dim() == 2 && b.dim() == 2, "kopt_wq: 2D CUDA inputs");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(0) == 1, "kopt_wq: a row-major, b column-major");
  TORCH_CHECK(a.size(1) == b.size(0), "kopt_wq: shape mismatch");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn, "kopt_wq: fp8");
  TORCH_CHECK(scales_a.numel() == a.size(0) && scales_b.numel() == b.size(1), "kopt_wq: per-row/per-col scales");
  TORCH_CHECK(scales_a.is_contiguous() && scales_b.is_contiguous(), "kopt_wq: contiguous scales");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32 && scales_b.scalar_type() == torch::kFloat32, "kopt_wq: fp32");
  torch::Tensor out = torch::empty({a.size(0), b.size(1)}, a.options().dtype(torch::kBFloat16));
  cutlass_gemm_caller_sm90_fp8<Gemm>(out, a, b, scales_a, scales_b);
  return out;
}
