#include <cstdlib>
#include <cassert>

#include <cute/tensor.hpp>

#include "flash_attention.h"
#include "type_util.h"
#include "cute_ext/boundary_check_pred.h"
#include "cute_ext/algorithms.h"

namespace cuda_device_functions {

using namespace cute;
using namespace cute_ext;

template <typename Src>
__device__
static constexpr inline auto ExtractFromBroadcastable(const int leading_index, Src&& src) {
  using ActualSrc = remove_cv_t<remove_reference_t<Src>>;
  static_assert(!is_tuple<ActualSrc>::value, "tuple type is not supported");

  constexpr auto rank = is_tensor<ActualSrc>::value ? decltype(cute::rank(src))::value : 0;

  if constexpr (rank == 0) {
    return src;
  } else if constexpr (rank == 1) {
    if constexpr (is_constant<1, decltype(size(src))>::value) {
      return src(Int<0>{});
    } else {
      return src(leading_index);
    }
  } else {
    const auto grouped_src = group_modes<1,rank>(src);
    constexpr auto expander = tuple_repeat<rank-1>(_);
    if constexpr (is_constant<1, decltype(size<0>(grouped_src))>::value) {
      return grouped_src(Int<0>{}, expander);
    } else {
      return grouped_src(leading_index, expander);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

template <typename ApplyingPredicator,
          typename DstPredicator,
          typename DstTensor,
          typename Func,
          typename... ArgTensors>
__device__
static inline void ApplyFunc(
                    const ApplyingPredicator& applying_predicator,
                    const DstPredicator& dst_predicator,
                    DstTensor& dst_tensor,
                    const typename DstTensor::value_type reset_value,
                    const Func& func,
                    const ArgTensors&... arg_tensors) {

  static_assert(decltype(rank(dst_tensor))::value == 2, "The rank of dst_tensor should be 2");

  static_assert((
                  (decltype(rank(arg_tensors))::value >= 0 && decltype(rank(arg_tensors))::value <= 2)
                && ...), "The ranks of all arg_tensors should be >= 0 and <= 2");

  CUTE_UNROLL
  for (int rest_row = 0; rest_row < size<0>(dst_tensor); ++rest_row) {

    const auto arg_tensor_tuple = make_tuple(ExtractFromBroadcastable(rest_row, arg_tensors)...);

    CUTE_UNROLL
    for (int rest_column = 0; rest_column < size<1>(dst_tensor); ++rest_column) {
      const auto rest_coords = make_coord(rest_row, rest_column);

      if (dst_predicator(rest_coords)) {
        typename DstTensor::value_type& current_value = dst_tensor(rest_coords);

        if (applying_predicator(rest_coords)) {

          const auto proxy_func = [&func, &current_value, rest_column](auto&&... args) {
            return func(current_value, ExtractFromBroadcastable(rest_column, static_cast<decltype(args)&&>(args))...);
          };

          current_value = apply(arg_tensor_tuple, proxy_func);
        }
        else
          current_value = reset_value;

      }
    }
  }
}

template <bool SaveApplied=false,
          typename ThreadLayout,
          typename ReductionPredicator,
          typename SrcDstPredicator,
          typename SrcDstTensor,
          typename ReducedResultTensor,
          typename ReductionFunc,
          typename ApplyingFunc,
          typename... ArgTensors>
__device__
static inline void ReduceAlongMode1WithWarp(
                    const ThreadLayout& thread_layout,
                    const ReductionPredicator& reduction_predicator,
                    const SrcDstPredicator& src_dst_predicator,
                    SrcDstTensor& src_dst_tensor,
                    ReducedResultTensor&& reduced_result_tensor,
                    const typename SrcDstTensor::value_type& reset_value,
                    const ReductionFunc& reduction_func,
                    const ApplyingFunc& applying_func = nullptr,
                    const ArgTensors&... arg_tensors) {


  using ValueType = typename SrcDstTensor::value_type;
  static_assert(std::is_same<typename std::remove_reference_t<std::remove_cv_t<ReducedResultTensor>>::value_type, ValueType>::value,
                "The value_type of src_dst_tensor and reduced_result_tensor should be the same");


  static_assert(decltype(rank(src_dst_tensor))::value == 2, "The rank of src_dst_tensor should be 2");
  static_assert(decltype(rank(thread_layout))::value == 2, "The rank of thread_layout should be 2");


  static_assert(decltype(rank(reduced_result_tensor))::value == 1,
                "The rank of reduced_result_tensor should be 1");

  static_assert(sizeof...(ArgTensors) == 0 || !std::is_null_pointer<ApplyingFunc>::value, "applying_func can not be nullptr when arg_tensors are provided");

  //static_assert((decltype(congruent(src_dst_tensor.shape(), arg_tensors.shape()))::value && ...), "The shape of all arg_tensors should be congruent with the one of src_dst_tensor");
  static_assert((
                  (decltype(rank(arg_tensors))::value >= 0 && decltype(rank(arg_tensors))::value <= 2)
                && ...), "The ranks of all arg_tensors should be >= 0 and <= 2");



  static constexpr auto SUB_WARP_SIZE = decltype(size<1>(thread_layout))::value;
  //static constexpr auto NUM_OF_SUB_WARPS = 32 / SUB_WARP_SIZE;

  const auto thread_coords = thread_layout.get_flat_coord(threadIdx.x);
  const auto thread_column_index = get<1>(thread_coords);
  //const auto sub_warp_offset = (thread_row_index % NUM_OF_SUB_WARPS) * SUB_WARP_SIZE;
  //const auto thread_index_in_warp = thread_column_index + sub_warp_offset;
  //const auto sub_warp_mask = static_cast<uint32_t>((uint64_t(1) << SUB_WARP_SIZE) - 1) << sub_warp_offset;


  CUTE_UNROLL
  for (int row_index = 0; row_index < size<0>(src_dst_tensor); ++row_index) {
    ValueType current_value = reset_value;

    const auto arg_tensor_tuple = make_tuple(ExtractFromBroadcastable(row_index, arg_tensors)...);

    CUTE_UNROLL
    for (int column_index = 0; column_index < size<1>(src_dst_tensor); ++column_index) {
      const auto rest_coords = make_coord(row_index, column_index);

      if (src_dst_predicator(rest_coords)) {
        ValueType& src_dst = src_dst_tensor(rest_coords);

        if (reduction_predicator(rest_coords)) {
          ValueType temp;
          if constexpr (!std::is_same<ApplyingFunc, decltype(nullptr)>::value) {
            const auto proxy_func = [&applying_func, &src_dst, column_index](auto&&... args) {
              return applying_func(src_dst, ExtractFromBroadcastable(column_index, std::forward<decltype(args)>(args))...);
            };
            temp = apply(arg_tensor_tuple, proxy_func);
            //temp = applying_func(src_dst, arg_tensors(rest_coords)...);
            if constexpr (SaveApplied)
              src_dst = temp;
          }
          else
            temp = src_dst;

          current_value = reduction_func(current_value, temp);
        }
        else
          src_dst = reset_value;
      }

    }

    // Reduction within a warp
    CUTE_UNROLL
    for (uint32_t reduction_step = SUB_WARP_SIZE >> 1; reduction_step > 0; reduction_step >>= 1) {
      current_value = reduction_func(current_value, __shfl_down_sync(0xFFFFFFFF, current_value, reduction_step, SUB_WARP_SIZE));
    }

    //current_value = __shfl_sync(0xFFFFFFFF, current_value, 0, SUB_WARP_SIZE);

    if (src_dst_predicator(row_index, Int<0>{})) {
      if (thread_column_index == 0) {
        reduced_result_tensor(row_index) = current_value;
      }
    }
  }

}

template <typename CopyOp,
          typename CopyPredicatorA, typename TensorA,
          typename CopyPredicatorB, typename TensorB,
          typename CopyPredicatorC, typename TensorC,
          typename PostFunc,
          typename... ArgTensors>
__device__
static inline void CopyToRegisterAndGEMM(
  const CopyOp& copy_op,
  const CopyPredicatorA& predicator_a, const TensorA& tensor_a,
  const CopyPredicatorB& predicator_b, const TensorB& tensor_b,
  const CopyPredicatorC& predicator_c, TensorC&& tensor_c,
  const PostFunc& post_func,
  const ArgTensors&... arg_tensors
  ) {

  // Shape:
  // tensor_a: ((block_m, block_k), (rest_m, rest_k))
  // tensor_b: ((block_n, block_k), (rest_n, rest_k))
  // tensor_c: ((block_m, block_n), (rest_m, rest_n))
  //

  using ActualTensorC = std::remove_reference_t<std::remove_cv_t<TensorC>>;

  using ValueType = typename ActualTensorC::value_type;
  static_assert(std::is_same<typename TensorA::value_type, typename TensorB::value_type>::value
                && std::is_same<typename TensorB::value_type, ValueType>::value,
                "tensor a/b/c should have the same value_type");

  static_assert(decltype(rank(tensor_a))::value == 2
                && decltype(rank<0>(tensor_a))::value == 2 && decltype(rank<1>(tensor_a))::value == 2,
                "tensor_a should be of the shape of ((block_m, block_k), (rest_m, rest_k))");
  static_assert(decltype(rank(tensor_b))::value == 2
                && decltype(rank<0>(tensor_b))::value == 2 && decltype(rank<1>(tensor_b))::value == 2,
                "tensor_b should be of the shape of ((block_n, block_k), (rest_n, rest_k))");
  static_assert(decltype(rank(tensor_c))::value == 2
                && decltype(rank<0>(tensor_c))::value == 2 && decltype(rank<1>(tensor_c))::value == 2,
                "tensor_c should be of the shape of ((block_m, block_n), (rest_m, rest_n))");

  static_assert(((decltype(rank(arg_tensors))::value == 2) && ...), "The ranks of all arg_tensors should be 2");


  static constexpr auto read_value_from_arg_tensor_block = [](const int row_index, const int column_index, const auto& block_src) constexpr -> auto {
    using Rank = decltype(rank(block_src));
    if constexpr (Rank::value == 1) {
      if constexpr (is_constant<1, decltype(size(block_src))>::value)
        return block_src(Int<0>{});
      else
        return block_src(row_index);
    }
    else if constexpr (Rank::value == 2) {
      return block_src(row_index, column_index);
    }
    else {
      static_assert(Rank::value > 0 && Rank::value <= 2, "The rank of block_src can only be 1 or 2");
    }
    CUTE_GCC_UNREACHABLE;
  };

  static constexpr auto extract_block_from_arg_tensor = [](const int rest_m_index, const int rest_n_index, const auto& src) constexpr -> auto {
    using Rank = decltype(rank<1>(src));
    constexpr auto rank0_expander = tuple_repeat<decltype(rank<0>(src))::value>(_);
    if constexpr (Rank::value == 1) {
      if constexpr (is_constant<1, decltype(size<1>(src))>::value)
        return src(rank0_expander, Int<0>{});
      else
        return src(rank0_expander, rest_m_index);
    }
    else if constexpr (Rank::value == 2) {
      return src(rank0_expander, make_coord(rest_m_index, rest_n_index));
    }
    else {
      static_assert(Rank::value > 0 && Rank::value <= 2, "The rank of src can only be 1 or 2");
    }
    CUTE_GCC_UNREACHABLE;
  };


  // Shape:
  //
  // register_a: block shape of tensor_a
  // register_b: block shape of tensor_b
  // register_c: block shape of tensor_c

  //static constexpr TrivialPredTensor always_trues{};

  Tensor register_a = make_tensor<ValueType>(get<0>(tensor_a.shape()));
  Tensor register_b = make_tensor<ValueType>(get<0>(tensor_b.shape()));
  Tensor register_c = make_tensor<ValueType>(get<0>(tensor_c.shape()));


  CUTE_UNROLL
  for (int rest_m_index = 0; rest_m_index < size<1, 0>(tensor_c); ++rest_m_index) {

    CUTE_UNROLL
    for (int rest_n_index = 0; rest_n_index < size<1, 1>(tensor_c); ++rest_n_index) {
      const auto rest_coords_c = make_coord(rest_m_index, rest_n_index);
      auto block_c = tensor_c(make_coord(_, _), rest_coords_c);
      const auto block_pred_c = predicator_c(make_coord(_, _), rest_coords_c);


      const auto arg_tensor_tuple = make_tuple(extract_block_from_arg_tensor(rest_m_index, rest_n_index, arg_tensors)...);

      clear(register_c);

      CUTE_UNROLL
      for (int rest_k_index = 0; rest_k_index < size<1, 1>(tensor_a); ++rest_k_index) {

        const auto rest_coords_a = make_coord(rest_m_index, rest_k_index);
        const auto rest_coords_b = make_coord(rest_n_index, rest_k_index);
        const auto block_a = tensor_a(make_coord(_, _), rest_coords_a);
        const auto block_pred_a = predicator_a(make_coord(_, _), rest_coords_a);
        const auto block_b = tensor_b(make_coord(_, _), rest_coords_b);
        const auto block_pred_b = predicator_b(make_coord(_, _), rest_coords_b);

        // Looping once instead of twice when copying data from tensor_a and tensor_b,
        // to reduce some execution time
        constexpr auto max_copying_size = size(block_a) > size(block_b) ? size(block_a) : size(block_b);
        CUTE_UNROLL
        for (int i = 0; i < max_copying_size; ++i) {
          if (i < decltype(size(block_a))::value) {
            if (block_pred_a(i))
              copy_op.copy(block_a(i), register_a(i));
            else
              register_a(i) = ValueType(0);
          }
          if (i < decltype(size(block_b))::value) {
            if (block_pred_b(i))
              copy_op.copy(block_b(i), register_b(i));
            else
              register_b(i) = ValueType(0);
          }
        }

        //CopyOrFill(copy_op, predicator_a(make_coord(_, _), rest_coords_a), tensor_a(make_coord(_, _), rest_coords_a), always_trues, register_a, 0);
        //CopyOrFill(copy_op, predicator_b(make_coord(_, _), rest_coords_b), tensor_b(make_coord(_, _), rest_coords_b), always_trues, register_b, 0);

        gemm(register_a, register_b, register_c);
      }

      CUTE_UNROLL
      for (int row_index = 0; row_index < decltype(size<0>(register_c))::value; ++row_index) {
        CUTE_UNROLL
        for (int column_index = 0; column_index < decltype(size<1>(register_c))::value; ++column_index) {
          if (block_pred_c(row_index, column_index)) {

            const auto proxy_post_func = [&post_func,
                                        &result = register_c(row_index, column_index),
                                        &c = block_c(row_index, column_index),
                                        row_index, column_index](auto&&... args) {
              return post_func(result, c, read_value_from_arg_tensor_block(row_index, column_index, std::forward<decltype(args)>(args))...);
            };

            block_c(row_index, column_index) = apply(arg_tensor_tuple, proxy_post_func);
          }
        }
      }
    }
  }

}

template <typename Tensor,
          typename ThreadLayout>
__device__
static constexpr inline auto PartitionPerThreadLayout(
  Tensor&& tensor, const ThreadLayout& thread_layout) {
  const auto thread_coord = thread_layout.get_flat_coord(threadIdx.x);
  return outer_partition(tensor, thread_layout.shape(), thread_coord);
}

#if 0
template <size_t AsDims, typename SrcTensor>
class TensorAs {
 public:

  __device__
  TensorAs(SrcTensor&& src)
  : _src(static_cast<SrcTensor&&>(src)) {
  }

  template <typename MainCoord, typename... RestCoords>
  __device__
  inline constexpr auto operator()(const MainCoord& main_coord, const RestCoords&... coords) const {
    if constexpr (is_tuple<MainCoord>::value)
      return _src(take<0, decltype(rank(_src))::value>(main_coord));
    else
      return _src(main_coord);
  }

  __device__
  inline constexpr auto shape() const {
    return append<AsDims>(_src.shape(), Int<1>{});
  }

 private:
  SrcTensor _src;
};

template <size_t AsDims, typename SrcTensor>
__device__
static inline constexpr auto ViewTensorAs(SrcTensor&& src) {
  return TensorAs<AsDims, SrcTensor>(static_cast<SrcTensor&&>(src));
}
#endif






#if 0
template <typename T, typename ReferenceSeqShape>
__device__
static inline constexpr auto GetAlibiParameters(const ReferenceSeqShape& reference_seq_shape, const T log_alibi_slope_base) {
  constexpr auto head_ch_index = decltype(rank(reference_seq_shape))::value - 1;
  const auto number_of_heads = get<head_ch_index>(reference_seq_shape);
  const int head_index = blockIdx.y % number_of_heads;
  const T alibi_slope = (log_alibi_slope_base > TypeUtil<T>::GetNegInfApprox())
                        ? TypeUtil<T>::Exp(static_cast<T>(head_index+1) * log_alibi_slope_base)
                        : T(0);

  return make_tuple(alibi_slope, size(take<0,head_ch_index>(reference_seq_shape)));
}
#endif


template <
          typename T, typename L_T,
          typename Q_Layout, typename Q_TileShape, typename Q_SM_Layout,
          typename K_Layout, typename K_TileShape, typename K_SM_Layout,
          typename V_Layout, typename V_TileShape, typename V_SM_Layout,
          typename O_Layout, typename O_TileShape,
          typename l_m_Layout, typename l_m_TileShape, typename l_m_SM_Layout,
          typename P_S_SM_Layout,
          typename ReferenceSeqShape,
          typename Q_SeqOrderMap, typename K_SeqOrderMap, typename AttentionPolicy>
__global__
__launch_bounds__(cuda_launch::KernelConfig<T>::NUM_OF_THREADS)
static void ForwardImpl(
    const T* Q,
    const Q_Layout Q_layout, const Q_TileShape Q_tile_shape, const Q_SM_Layout Q_sm_layout,
    const T* K,
    const K_Layout K_layout, const K_TileShape K_tile_shape, const K_SM_Layout K_sm_layout,
    const T* V,
    const V_Layout V_layout, const V_TileShape V_tile_shape, const V_SM_Layout V_sm_layout,
    volatile T* O,
    const O_Layout O_layout, const O_TileShape O_tile_shape,
    volatile L_T* l, volatile T* m,
    const l_m_Layout l_m_layout, const l_m_TileShape l_m_tile_shape, const l_m_SM_Layout l_m_sm_layout,
    const P_S_SM_Layout P_S_sm_layout,
    const T dot_scaler,
    volatile uint32_t* Br_occupancy,
    const ReferenceSeqShape reference_seq_shape,
    const Q_SeqOrderMap Q_seq_order_map, const K_SeqOrderMap K_seq_order_map, const AttentionPolicy attention_policy) {

  static constexpr typename cuda_launch::KernelConfig<T>::Mode0MajorThreadLayout MODE0_MAJOR_THREAD_LAYOUT{};
  static constexpr typename cuda_launch::KernelConfig<T>::Mode1MajorThreadLayout MODE1_MAJOR_THREAD_LAYOUT{};

  //
  // Tensors wrapping different portions of the shared memory buffer allocated dynamically
  //

  extern __shared__ char sm_buffer[];
  char* moving_head = sm_buffer;

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  // (Br, Bc)
  Tensor P_S_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), P_S_sm_layout);
  moving_head += cosize(P_S_sm_layout) * sizeof(T);
#else
  static constexpr typename cuda_launch::KernelConfig<T>::SmSwizzler SM_SWIZZLER;

  // Note that for address spaces being swizzled (like P_S_sm),
  // the start addresses of them need to be aligned with a transaction boundary
  // (i.e. 128-byte aligned), so that the designated swizzling configuration works correctly.
  //
  // Therefore, they are all allocated first to exploit the fact that
  // shared memory buffers are naturally 128-byte aligned, and
  // the size of each swizzled memery area is a multiple of the size of a transaction by design

  // (Br, Bc)
  Tensor P_S_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head), SM_SWIZZLER), P_S_sm_layout);
  moving_head += cosize(P_S_sm_layout) * sizeof(T);
#endif

  // (Br, d)
  Tensor Q_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), Q_sm_layout);
  moving_head += cosize(Q_sm_layout) * sizeof(T);
  // (Bc, d)
  Tensor K_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), K_sm_layout);
  moving_head += cosize(K_sm_layout) * sizeof(T);
  // (Bc, v_d)
  Tensor V_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), V_sm_layout);
  // (v_d, Bc)
  Tensor V_sm_swap = make_tensor(V_sm.data(), make_layout(get<1>(V_sm_layout), get<0>(V_sm_layout)));
  moving_head += cosize(V_sm_layout) * sizeof(T);

  // (Br, 1)
  const auto l_m_memory_size = cosize(l_m_sm_layout) * sizeof(T);
  Tensor l_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), l_m_sm_layout);
  //Tensor l_sm_T = recast<T>(l_sm);
  moving_head += l_m_memory_size;
  Tensor m_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), l_m_sm_layout);
  moving_head += l_m_memory_size;

  //
  // Tensors wrapping thier respective global memory pointers
  //

  // (q, d, b)
  const Tensor full_Q = make_tensor(make_gmem_ptr(Q), Q_layout);
  // (k, d, b)
  const Tensor full_K = make_tensor(make_gmem_ptr(K), K_layout);
  // (k, v_d, b)
  const Tensor full_V = make_tensor(make_gmem_ptr(V), V_layout);
  // (q, v_d, b)
  Tensor full_O = make_tensor(make_gmem_ptr(O), O_layout);
  // (q, 1, b)
  Tensor full_l = make_tensor(make_gmem_ptr(l), l_m_layout);
  Tensor full_m = make_tensor(make_gmem_ptr(m), l_m_layout);

  //
  // Identity maps providing coordinates used by predicators for range/boundary checks
  //

  const Tensor full_Q_map = make_identity_tensor(full_Q.shape());
  const Tensor full_K_map = make_identity_tensor(full_K.shape());
  const Tensor full_V_map = make_identity_tensor(full_V.shape());
  const Tensor full_O_map = make_identity_tensor(full_O.shape());


  // This map will be shared across l_sm and m_sm as they all have the same layout
  const Tensor full_l_m_map = make_identity_tensor(full_l.shape());

  const Tensor Q_sm_map = make_identity_tensor(Q_sm.shape());
  const Tensor K_sm_map = make_identity_tensor(K_sm.shape());
  const Tensor V_sm_map = make_identity_tensor(V_sm.shape());
  const Tensor V_sm_swap_map = make_identity_tensor(V_sm_swap.shape());
  //const Tensor O_sm_map = make_identity_tensor(O_sm.shape());
  const Tensor P_S_sm_map = make_identity_tensor(P_S_sm.shape());

  // This map will be shared between l_sm and m_sm as they all have the same layout
  const Tensor l_m_sm_map = make_identity_tensor(l_sm.shape());

  //
  // Tiling tensors wrapping global memories;
  // blockIdx.x indices select a Bc section,
  // while blockIdx.y selects a batch
  //

  // (Br, d, # of Br sections)
  const auto Q_O_rest_coords =  make_coord(_, Int<0>{}, blockIdx.y);
  const Tensor Q_tile = local_tile(full_Q, Q_tile_shape, Q_O_rest_coords)(_, _, Int<0>{}, _);
  const Tensor Q_tile_map = local_tile(full_Q_map, Q_tile_shape, Q_O_rest_coords)(_, _, Int<0>{}, _);


  // (Bc, d)
  const auto K_V_rest_coords = make_coord(blockIdx.x, Int<0>{}, blockIdx.y);
  const Tensor K_tile = local_tile(full_K, K_tile_shape, K_V_rest_coords)(_, _, Int<0>{});
  const Tensor K_tile_map = local_tile(full_K_map, K_tile_shape, K_V_rest_coords)(_, _, Int<0>{});

  // (Bc, v_d)
  const Tensor V_tile = local_tile(full_V, V_tile_shape, K_V_rest_coords)(_, _, Int<0>{});
  const Tensor V_tile_map = local_tile(full_V_map, V_tile_shape, K_V_rest_coords)(_, _, Int<0>{});

  // (Br, v_d, # of Br sections)
  Tensor O_tile = local_tile(full_O, O_tile_shape, Q_O_rest_coords)(_, _, Int<0>{}, _);
  const Tensor O_tile_map = local_tile(full_O_map, O_tile_shape, Q_O_rest_coords)(_, _, Int<0>{}, _);

  // Br_occupancy is be set up to a tiled structure,
  // so it only takes a correct layout and indexing to access the target tile
  // (# of Br sections)
  Tensor Br_occupancy_tile = make_tensor(make_gmem_ptr(Br_occupancy),
                                    make_layout(make_shape(size<2>(O_tile), size<2>(full_O))))(_, blockIdx.y);

  // (Br, 1, # of Br sections)
  const auto l_m_rest_coords = make_coord(_, Int<0>{}, blockIdx.y);
  Tensor l_tile = local_tile(full_l, l_m_tile_shape, l_m_rest_coords)(_, _, Int<0>{}, _);
  Tensor m_tile = local_tile(full_m, l_m_tile_shape, l_m_rest_coords)(_, _, Int<0>{}, _);
  const Tensor l_m_tile_map = local_tile(full_l_m_map, l_m_tile_shape, l_m_rest_coords)(_, _, Int<0>{}, _);




  //
  // Partitioning tensors with thread layouts
  //

  constexpr auto M0M_MODE0_THREAD_LAYOUT = select<0>(MODE0_MAJOR_THREAD_LAYOUT);
  constexpr auto M0M_MODE1_THREAD_LAYOUT = select<1>(MODE0_MAJOR_THREAD_LAYOUT);

  constexpr auto M1M_MODE0_THREAD_LAYOUT = select<0>(MODE1_MAJOR_THREAD_LAYOUT);
  constexpr auto M1M_MODE1_THREAD_LAYOUT = select<1>(MODE1_MAJOR_THREAD_LAYOUT);

  //
  // Q
  //

  // (# of thread groups along Br, # of thread groups along d, # of Br sections)
  const Tensor Q_tile_m0m_partitioned = PartitionPerThreadLayout(Q_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor Q_tile_map_m0m_partitioned = PartitionPerThreadLayout(Q_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, # of thread groups along d)
  Tensor Q_sm_m0m_partitioned = PartitionPerThreadLayout(Q_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor Q_sm_map_m0m_partitioned = PartitionPerThreadLayout(Q_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, d)
  const Tensor Q_sm_m1m_mode0_partitioned = PartitionPerThreadLayout(Q_sm, M1M_MODE0_THREAD_LAYOUT);
  const Tensor Q_sm_map_m1m_mode0_partitioned = PartitionPerThreadLayout(Q_sm_map, M1M_MODE0_THREAD_LAYOUT);


  //
  // K
  //

  // (# of thread groups along Bc, # of thread groups along d)
  const Tensor K_tile_m0m_partitioned = PartitionPerThreadLayout(K_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor K_tile_map_m0m_partitioned = PartitionPerThreadLayout(K_tile_map, MODE0_MAJOR_THREAD_LAYOUT);
  Tensor K_sm_m0m_partitioned = PartitionPerThreadLayout(K_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor K_sm_map_m0m_partitioned = PartitionPerThreadLayout(K_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Bc, d)
  const Tensor K_sm_m1m_mode1_partitioned = PartitionPerThreadLayout(K_sm, M1M_MODE1_THREAD_LAYOUT);
  const Tensor K_sm_map_m1m_mode1_partitioned = PartitionPerThreadLayout(K_sm_map, M1M_MODE1_THREAD_LAYOUT);


  //
  // V
  //

  // (# of thread groups along Bc, # of thread groups along v_d)
  const Tensor V_tile_m0m_partitioned = PartitionPerThreadLayout(V_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor V_tile_map_m0m_partitioned = PartitionPerThreadLayout(V_tile_map, MODE0_MAJOR_THREAD_LAYOUT);
  Tensor V_sm_m0m_partitioned = PartitionPerThreadLayout(V_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor V_sm_map_m0m_partitioned = PartitionPerThreadLayout(V_sm_map, MODE0_MAJOR_THREAD_LAYOUT);


  // (# of thread groups along v_d, # of thread groups along Bc)
  const Tensor V_sm_swap_m0m_mode1_partitioned = PartitionPerThreadLayout(V_sm_swap, M0M_MODE1_THREAD_LAYOUT);
  const Tensor V_sm_swap_map_m0m_mode1_partitioned = PartitionPerThreadLayout(V_sm_swap_map, M0M_MODE1_THREAD_LAYOUT);

  //
  // l / m
  //
  // Note that the dimension 1 (indices start from 0) of l/m is just 1;
  // therefore that dimension can be indexed out directly

  // (# of thread groups along Br, # of Br sections)
  const Tensor l_tile_m0m_partitioned_col0 = PartitionPerThreadLayout(l_tile, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);
  const Tensor m_tile_m0m_partitioned_col0 = PartitionPerThreadLayout(m_tile, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);

  const Tensor l_m_tile_map_m0m_partitioned_col0 = PartitionPerThreadLayout(l_m_tile_map, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);


  // (# of thread groups along Br)
  Tensor l_sm_m1m_mode0_partitioned_col0 = PartitionPerThreadLayout(l_sm, M1M_MODE0_THREAD_LAYOUT)(_, Int<0>{});
  // the memory of l_sm is reused to store weights of O
  const Tensor O_weight_sm_m0m_mode0_partitioned_col0 = PartitionPerThreadLayout(l_sm, M0M_MODE0_THREAD_LAYOUT)(_, Int<0>{});

  Tensor m_sm_m1m_mode0_partitioned_col0 = PartitionPerThreadLayout(m_sm, M1M_MODE0_THREAD_LAYOUT)(_, Int<0>{});
  // The memory of m_sm is reused to store weights of P_S
  const auto& P_S_weight_sm_m1m_mode0_partitioned_col0 = m_sm_m1m_mode0_partitioned_col0;

  Tensor l_sm_m0m_partitioned_col0 = PartitionPerThreadLayout(l_sm, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});
  Tensor m_sm_m0m_partitioned_col0 = PartitionPerThreadLayout(m_sm, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});

  const Tensor l_m_sm_map_m0m_partitioned_col0 = PartitionPerThreadLayout(l_m_sm_map, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});

  //
  // O
  //

  // (# of thread groups along Br, # of thread groups along v_d, # of Br sections)
  Tensor O_tile_m0m_partitioned = PartitionPerThreadLayout(O_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor O_tile_map_m0m_partitioned = PartitionPerThreadLayout(O_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  //
  // P_S
  //

  // (# of thread groups along Br, # of thread groups along Bc)
  Tensor P_S_sm_m0m_partitioned = PartitionPerThreadLayout(P_S_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor P_S_sm_map_m0m_partitioned = PartitionPerThreadLayout(P_S_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor P_S_sm_m1m_partitioned = PartitionPerThreadLayout(P_S_sm, MODE1_MAJOR_THREAD_LAYOUT);
  const Tensor P_S_sm_map_m1m_partitioned = PartitionPerThreadLayout(P_S_sm_map, MODE1_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, Bc)
  const Tensor P_S_sm_m0m_mode0_partitioned = PartitionPerThreadLayout(P_S_sm, M0M_MODE0_THREAD_LAYOUT);
  const Tensor P_S_sm_map_m0m_mode0_partitioned = PartitionPerThreadLayout(P_S_sm_map, M0M_MODE0_THREAD_LAYOUT);


  //
  // boundary_check predicators which can be created in advance
  //

  const auto Q_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(Q_sm_map_m0m_partitioned, Q_sm.shape());
  const auto K_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(K_sm_map_m0m_partitioned, K_sm.shape());
  const auto V_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(V_sm_map_m0m_partitioned, V_sm.shape());

  // l_sm and m_sm should have the same shape, so they share the same check
  const auto l_m_sm_map_m0m_partitioned_col0_boundary_check = GetBoundaryCheckPred<0, 1>(l_m_sm_map_m0m_partitioned_col0, l_sm.shape());
  const auto P_S_sm_map_m1m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(P_S_sm_map_m1m_partitioned, P_S_sm.shape());

  #if 0
  //
  // alibi
  //

  const auto alibi_params = GetAlibiParameters(reference_seq_shape, log_alibi_slope_base);
  const T alibi_slope = get<0>(alibi_params);
  const auto alibi_cal_normaliser = static_cast<T>(get<1>(alibi_params));

  const auto normalised_diff_squared = [alibi_cal_normaliser](const auto& a, const auto& b) {
    const auto normalised = static_cast<decltype(alibi_cal_normaliser)>(a-b) / alibi_cal_normaliser;
    return normalised * normalised;
  };
  #endif

  //
  // Various functors/lambda functions
  //

  //static constexpr Copy_Atom<UniversalCopy<T>, T> copy_atom{};

  static constexpr UniversalCopy<T> copy_op{};

  const T masking_value = TypeUtil<T>::GetNegInfApprox();

  static constexpr TrivialPredTensor always_trues{};

  static constexpr auto max_func = [](const T& current, const T& next) constexpr {
                                          //return current < next ? next : current;
                                          return TypeUtil<T>::Max(current, next);
                                        };

  static constexpr multiplies multiplication_func{};
  static constexpr plus addition_func{};



  const auto numerator_calc_func = [masking_value](const T& current_value, const T& broadcast_value) {
                                          if (current_value <= masking_value)
                                            return T(0);
                                          else
                                            return TypeUtil<T>::Exp(current_value - broadcast_value);
                                        };

  static constexpr auto identity_post_func = [](const T& current_value, const T&) constexpr {
                                                      return current_value;
                                                    };
  static constexpr auto weighted_sum_pos_func = [](const T& current_value, const volatile T& existing_value, const T& weight) constexpr {
                                                      return current_value + const_cast<const T&>(existing_value) * weight;
                                                    };

  //
  // Additional register block partitioning for GEMM0/GEMM1
  //

  // Block sizes of gemm "O = M @ N" below are chosen based on the thread layout used
  // and my test results, so they might not be optimal under different threading configurations
  // and/or with GPUs of different capabilities
  static constexpr auto GEMM0_M_REGISTER_BLOCK_TILER = make_shape(Int<1>{}, Int<4>{});
  static constexpr auto GEMM0_N_REGISTER_BLOCK_TILER = make_shape(Int<4>{}, Int<4>{});
  static constexpr auto GEMM0_O_REGISTER_BLOCK_TILER = make_shape(get<0>(GEMM0_M_REGISTER_BLOCK_TILER), get<0>(GEMM0_N_REGISTER_BLOCK_TILER));

  static constexpr auto GEMM1_M_REGISTER_BLOCK_TILER = make_shape(Int<4>{}, Int<4>{});
  static constexpr auto GEMM1_N_REGISTER_BLOCK_TILER = make_shape(Int<2>{}, Int<4>{});
  static constexpr auto GEMM1_O_REGISTER_BLOCK_TILER = make_shape(get<0>(GEMM1_M_REGISTER_BLOCK_TILER), get<0>(GEMM1_N_REGISTER_BLOCK_TILER));


  // (GEMM1_M_REGISTER_BLOCK_SHAPE, (rest dimension associated with Br, rest dimension associated with d))
  const Tensor Q_sm_m1m_mode0_partitioned_gemm_block = zipped_divide(Q_sm_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);
  const Tensor Q_sm_map_m1m_mode0_partitioned_gemm_block = zipped_divide(Q_sm_map_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);

  // (GEMM1_N_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with d))
  const Tensor K_sm_m1m_mode1_partitioned_gemm_block = zipped_divide(K_sm_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);
  const Tensor K_sm_map_m1m_mode1_partitioned_gemm_block = zipped_divide(K_sm_map_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);

  // (GEMM1_O_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with Bc))
  Tensor P_S_sm_m1m_partitioned_gemm_block = zipped_divide(P_S_sm_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);
  const Tensor P_S_sm_map_m1m_partitioned_gemm_block = zipped_divide(P_S_sm_map_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);

  const auto Q_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(Q_sm_map_m1m_mode0_partitioned_gemm_block, Q_sm.shape());
  const auto K_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(K_sm_map_m1m_mode1_partitioned_gemm_block, K_sm.shape());
  const auto P_S_sm_map_m1m_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(P_S_sm_map_m1m_partitioned_gemm_block, P_S_sm.shape());


  // (GEMM0_M_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with Bc))
  const Tensor P_S_sm_m0m_mode0_partitioned_gemm_block = zipped_divide(P_S_sm_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);
  const Tensor P_S_sm_map_m0m_mode0_partitioned_gemm_block = zipped_divide(P_S_sm_map_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);

  // (GEMM0_N_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with v_d))
  const Tensor V_sm_swap_m0m_mode1_partitioned_gemm_block = zipped_divide(V_sm_swap_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);
  const Tensor V_sm_swap_map_m0m_mode1_partitioned_gemm_block = zipped_divide(V_sm_swap_map_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);

  // (GEMM0_O_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with v_d), # of Br sections)
  Tensor O_tile_m0m_partitioned_gemm_block = group_modes<1,3>(zipped_divide(O_tile_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER)(_, repeat<3>(_)));
  const Tensor O_tile_map_m0m_partitioned_gemm_block = group_modes<1,3>(zipped_divide(O_tile_map_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER)(_, repeat<3>(_)));

  // (GEMM0_O_REGISTER_BLOCK_TILER[0:1], (rest dimension associated with Br))
  const Tensor O_weight_sm_m0m_mode0_partitioned_col0_gemm_block = zipped_divide(O_weight_sm_m0m_mode0_partitioned_col0, select<0>(GEMM0_O_REGISTER_BLOCK_TILER));

  const auto P_S_sm_map_m0m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(P_S_sm_map_m0m_mode0_partitioned_gemm_block, P_S_sm.shape());
  const auto V_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(V_sm_swap_map_m0m_mode1_partitioned_gemm_block, V_sm_swap.shape());


  //
  // Loading data and performing forward calculation
  //

  const auto first_K_index = blockIdx.x * get<0>(K_tile_shape);
  const auto last_K_index = (blockIdx.x+1) * get<0>(K_tile_shape) - 1;
  const auto bbox_min_for_K = make_tuple(min(first_K_index, get<0>(full_K.shape())-1), Int<0>{}, blockIdx.y);
  const auto bounded_last_K_index = min(first_K_index+get<0>(K_tile_shape)-1, get<0>(full_K.shape())-1);

  const auto max_order = size(reference_seq_shape) - 1;
  const auto min_k_order = get<0>(K_seq_order_map(first_K_index));
  const auto max_k_order = min(get<0>(K_seq_order_map(last_K_index)), max_order);

  // Loading target blocks of K and V into K_sm and V_sm respectively,
  // in a single loop rather than separate ones for each loading
  const auto K_tile_map_m0m_partitioned_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(K_tile_map_m0m_partitioned,
                                          bbox_min_for_K,
                                          make_tuple(bounded_last_K_index, get<1>(full_K.shape())-1, blockIdx.y));

  const auto V_tile_map_m0m_partitioned_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(V_tile_map_m0m_partitioned,
                                          bbox_min_for_K,
                                          make_tuple(bounded_last_K_index, get<1>(full_V.shape())-1, blockIdx.y));

  const auto max_size_of_K_V_sm_m0m_partitioning = max(size(K_sm_m0m_partitioned), size(V_sm_m0m_partitioned));

  CUTE_UNROLL
  for (int scanning_index = 0; scanning_index < max_size_of_K_V_sm_m0m_partitioning; ++scanning_index) {
    if (K_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
      if (K_tile_map_m0m_partitioned_boundary_check(scanning_index))
        copy_op.copy(K_tile_m0m_partitioned(scanning_index), K_sm_m0m_partitioned(scanning_index));
      else
        K_sm_m0m_partitioned(scanning_index) = T(0);
    }

    if (V_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
      if (V_tile_map_m0m_partitioned_boundary_check(scanning_index))
        copy_op.copy(V_tile_m0m_partitioned(scanning_index), V_sm_m0m_partitioned(scanning_index));
      else
        V_sm_m0m_partitioned(scanning_index) = T(0);
    }
  }



  // Having loops over Br sections processed by each thread block start at different indices,
  // to decrease the possiblity of racing for updating the same data sections assoicated with
  // the same Br index
  const int num_of_Br = size(Br_occupancy_tile);
  const int Br_index_start = blockIdx.x * max(1, num_of_Br / gridDim.x);

  CUTE_UNROLL
  for (int Br_scan = 0; Br_scan < num_of_Br; ++Br_scan) {
    int Br_index = (Br_index_start + Br_scan) % num_of_Br;

    const auto first_Q_index = Br_index * get<0>(Q_tile_shape);
    const auto last_Q_index = (Br_index+1) * get<0>(Q_tile_shape) - 1;

    // Early Checking to see if we can skip this iteration entirely.
    // For example, when doing causal attention, no need to process blocks of Q in which all examples are located before the current block of K
    if (attention_policy.IsSkipped(reference_seq_shape,
                                    get<0>(Q_seq_order_map(first_Q_index)), min(get<0>(Q_seq_order_map(last_Q_index)), max_order),
                                    min_k_order, max_k_order)) {
      continue;
    }

    const auto bbox_min_for_Q = make_tuple(min(first_Q_index, get<0>(full_Q.shape())-1), Int<0>{}, blockIdx.y);
    const auto bounded_last_Q_index = min(last_Q_index, get<0>(full_Q.shape())-1);

    // Since Q is globally constant during the whole process,
    // it is fine to load data from Q for this run of loops first,
    // then wait for the Br_occupancy_indicator to be cleared,
    // so that only one __syncthreads() is needed for these operations
    CopyOrFill(copy_op,
                  GetBoundaryCheckPred<0, 1, 2>(Q_tile_map_m0m_partitioned(_, _, Br_index),
                                    bbox_min_for_Q,
                                    make_tuple(bounded_last_Q_index, get<1>(full_Q.shape())-1, blockIdx.y)),
                  Q_tile_m0m_partitioned(_, _, Br_index),
                  Q_sm_map_m0m_partitioned_boundary_check,
                  Q_sm_m0m_partitioned,
                  0);


    // Only thread 0 checks if this block can proceed to process this Br section,
    // while others are waiting at the __syncthreads() below
    if (threadIdx.x == 0) {
      // Note that BrOccupancyValueType does not contain volatile qualifier
      using BrOccupancyValueType = typename decltype(Br_occupancy_tile)::value_type;
      BrOccupancyValueType* Br_occupancy_indicator = const_cast<BrOccupancyValueType*>(&Br_occupancy_tile(Br_index));
      while (atomicCAS(Br_occupancy_indicator, 0, 1)) ;
    }

    __syncthreads();

    // Computing Q K^T
    CopyToRegisterAndGEMM(
      copy_op,
      Q_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check,
      Q_sm_m1m_mode0_partitioned_gemm_block,
      K_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check,
      K_sm_m1m_mode1_partitioned_gemm_block,
      P_S_sm_map_m1m_partitioned_gemm_block_boundary_check,
      P_S_sm_m1m_partitioned_gemm_block,
      identity_post_func
    );



    const auto adjust_per_position = [Br_index, &Q_tile_map, &Q_layout, &K_tile_map, &K_layout,
      &reference_seq_shape, &Q_seq_order_map, &K_seq_order_map, &attention_policy,
      masking_value,
      dot_scaler] (const T& current_value, const auto& coords) {


      const auto q = get<0>(Q_tile_map(get<0>(coords), Int<0>{}, Br_index));
      const auto q_order = get<0>(Q_seq_order_map(q));
      const auto k = get<0>(K_tile_map(get<1>(coords), Int<0>{}));
      const auto k_order = get<0>(K_seq_order_map(k));


      if (q < size<0>(Q_layout) && k < size<0>(K_layout)
              && attention_policy.Check(reference_seq_shape, q_order, k_order)) {
        return current_value * dot_scaler;
      } else {
        return masking_value;
      }
    };


    // Reducing along mode 1 to find the maximum value, masking unallowed entries
    // indicated by the designated attention policy in the meantime

    ReduceAlongMode1WithWarp<true>(MODE1_MAJOR_THREAD_LAYOUT,
                      always_trues,
                      P_S_sm_map_m1m_partitioned_boundary_check,
                      P_S_sm_m1m_partitioned,
                      m_sm_m1m_mode0_partitioned_col0,
                      masking_value,
                      max_func,
                      adjust_per_position,
                      P_S_sm_map_m1m_partitioned);


    __syncwarp();


    // Reducing along mode 1 again to compute the denominator for normalisation,
    // updating each entry of P_S_sm to hold the corresponding numerator value in the meantime
    ReduceAlongMode1WithWarp<true>(
                  MODE1_MAJOR_THREAD_LAYOUT,
                  always_trues,
                  P_S_sm_map_m1m_partitioned_boundary_check,
                  P_S_sm_m1m_partitioned,
                  l_sm_m1m_mode0_partitioned_col0,
                  0,
                  addition_func,
                  numerator_calc_func,
                  m_sm_m1m_mode0_partitioned_col0);


    __syncthreads();


    const auto l_m_tile_map_m0m_partitioned_col0_boundary_check = GetBoundaryCheckPred<0, 1, 2>(l_m_tile_map_m0m_partitioned_col0(_, Br_index),
                                  bbox_min_for_Q,
                                  make_tuple(bounded_last_Q_index, get<1>(full_l.shape())-1, blockIdx.y));

    CUTE_UNROLL
    for (int rest_index = 0; rest_index < size(l_sm_m0m_partitioned_col0); ++rest_index) {
      if (l_m_sm_map_m0m_partitioned_col0_boundary_check(rest_index)) {

        T& l_tilde = l_sm_m0m_partitioned_col0(rest_index);
        T& m_tilde = m_sm_m0m_partitioned_col0(rest_index);


        if (l_m_tile_map_m0m_partitioned_col0_boundary_check(rest_index)) {
          volatile L_T& l_current = l_tile_m0m_partitioned_col0(rest_index, Br_index);
          volatile T& m_current = m_tile_m0m_partitioned_col0(rest_index, Br_index);

          T m_tilde_cache = m_tilde;

          L_T l_current_cache = l_current;
          T m_current_cache = m_current;

          L_T l_tilde_weight = L_T(0), weighted_l_current = l_current_cache;
          T new_m = m_current_cache;


          if (m_tilde_cache > masking_value) {
            new_m = TypeUtil<T>::Max(m_current_cache, m_tilde_cache);
            l_tilde_weight = TypeUtil<L_T>::Exp(static_cast<L_T>(m_tilde_cache - new_m));
            weighted_l_current *= TypeUtil<L_T>::Exp(static_cast<L_T>(m_current_cache - new_m));
          }


          l_current = l_current_cache = weighted_l_current + l_tilde_weight * static_cast<L_T>(l_tilde);
          m_current = new_m;

          // l_current_cache at this point has been updated to a new one;
          // also the memory of l_tilde is reused to store the weights for O,
          // while the memory of m_tilde is reused to store the weight for S_P
          if (new_m <= masking_value) {
            // In this case, we haven't seen any unmasked entries at the associated row
            // in the full attention matrix, and this implies there won't be any
            // contribution to the final outcome of that row from this block;
            // so the weight for O and S_P are set to 0 directly instead of
            // arithmetic calculation, as the latter might lead to numerical instability
            l_tilde = T(0);
            m_tilde = T(0);
          }
          else {
            l_tilde = static_cast<T>(weighted_l_current / l_current_cache);
            m_tilde = static_cast<T>(l_tilde_weight / l_current_cache);
          }


        }
        else {
          // In this case, this part of partitioned_l_m0m_sm/partitioned_m_m0m_sm is
          // outside the region of the global memory required to be processed;
          // to prevent their values, as well as others associated with them in later computation,
          // from accidentally contributing to the final outcome,
          // weights for O (stored in sub_l_sm via l_tilde) and S_P (stored in sub_m_sm via m_tilde)
          // are set to 0 here
          l_tilde = T(0);
          m_tilde = T(0);
        }
      }
    }


    __syncthreads();


    ApplyFunc(
            always_trues,
            P_S_sm_map_m1m_partitioned_boundary_check,
            P_S_sm_m1m_partitioned,
            0,
            multiplication_func,
            P_S_weight_sm_m1m_mode0_partitioned_col0
          );


    __syncthreads();

    CopyToRegisterAndGEMM(
      copy_op,
      P_S_sm_map_m0m_mode0_partitioned_gemm_block_boundary_check,
      P_S_sm_m0m_mode0_partitioned_gemm_block,
      V_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check,
      V_sm_swap_m0m_mode1_partitioned_gemm_block,
      GetBoundaryCheckPred<0, 1, 2>(O_tile_map_m0m_partitioned_gemm_block(_, _, Br_index),
                                    bbox_min_for_Q,
                                    make_tuple(bounded_last_Q_index, get<1>(full_O.shape())-1, blockIdx.y)),
      O_tile_m0m_partitioned_gemm_block(_, _, Br_index),
      weighted_sum_pos_func,
      O_weight_sm_m0m_mode0_partitioned_col0_gemm_block
    );

    // __threadfence() flushes all writes above back to L2/global memory,
    // and the following __syncthreads() makes sure that flushes triggered by all threads are done
    // before thread 0 goes on to reset the occupancy flag
    __threadfence();
    __syncthreads();
    // only thread 0 does the reset of the occupancy flag of this Br section while others go ahead
    if (threadIdx.x == 0) {
      Br_occupancy_tile(Br_index) = 0;
    }
  }
}

template <
          typename T, typename L_T,
          typename Q_DQ_Layout, typename Q_DQ_TileShape, typename Q_SM_Layout,
          typename K_DK_Layout, typename K_DK_TileShape, typename K_DK_SM_Layout,
          typename V_DV_Layout, typename V_DV_TileShape, typename V_DV_SM_Layout,
          typename O_DO_Layout, typename O_DO_TileShape, typename O_DO_SM_Layout,
          typename l_m_Layout, typename l_m_TileShape, typename l_m_SM_Layout,
          typename P_SM_Layout, typename S_SM_Layout,
          typename ReferenceSeqShape,
          typename Q_SeqOrderMap, typename K_SeqOrderMap,
          typename AttentionPolicy>
__global__
__launch_bounds__(cuda_launch::KernelConfig<T>::NUM_OF_THREADS)
static void BackwardImpl(
    const T* Q,
    const Q_DQ_Layout Q_dQ_layout, const Q_DQ_TileShape Q_dQ_tile_shape, const Q_SM_Layout Q_sm_layout,
    const T* K,
    const K_DK_Layout K_dK_layout, const K_DK_TileShape K_dK_tile_shape, const K_DK_SM_Layout K_dK_sm_layout,
    const T* V,
    const V_DV_Layout V_dV_layout, const V_DV_TileShape V_dV_tile_shape, const V_DV_SM_Layout V_dV_sm_layout,
    const T* O, const T* dO,
    const O_DO_Layout O_dO_layout, const O_DO_TileShape O_dO_tile_shape, const O_DO_SM_Layout O_dO_sm_layout,
    const L_T* l, const T* m, const l_m_Layout l_m_layout, const l_m_TileShape l_m_tile_shape, const l_m_SM_Layout l_m_sm_layout,
    volatile T* dQ, T* dK, T* dV,
    const P_SM_Layout P_sm_layout, const S_SM_Layout S_sm_layout,
    const T dot_scaler,
    volatile uint32_t* Br_occupancy,
    const ReferenceSeqShape reference_seq_shape,
    const Q_SeqOrderMap Q_seq_order_map, const K_SeqOrderMap K_seq_order_map, const AttentionPolicy attention_policy) {

  static constexpr typename cuda_launch::KernelConfig<T>::Mode0MajorThreadLayout MODE0_MAJOR_THREAD_LAYOUT{};
  static constexpr typename cuda_launch::KernelConfig<T>::Mode1MajorThreadLayout MODE1_MAJOR_THREAD_LAYOUT{};

  //
  // Tensors wrapping different portions of the shared memory buffer allocated dynamically
  //

  extern __shared__ char sm_buffer[];
  char* moving_head = sm_buffer;

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  // Since the size of L_T might be larger than T,
  // the space of l_sm is appropriated before others to avoid alignment errors

  // (Br, 1)
  Tensor l_sm = make_tensor(make_smem_ptr(reinterpret_cast<L_T*>(moving_head)), l_m_sm_layout);
  //Tensor l_sm_T = recast<T>(l_sm);
  moving_head += cosize(l_m_sm_layout) * sizeof(L_T);

  // (Br, v_d)
  Tensor dO_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), O_dO_sm_layout);
  moving_head += cosize(O_dO_sm_layout) * sizeof(T);

  // (Br, Bc)
  Tensor S_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), S_sm_layout);
  // Note that O_sm and S_sm share the same memory area, but occupy different column sizes
  // (therfore having different layouts), and the memory size reserved for them
  // should be the maximum of the two
  moving_head += max(cosize(S_sm_layout), cosize(O_dO_sm_layout)) * sizeof(T);

#else
  static constexpr typename cuda_launch::KernelConfig<T>::SmSwizzler SM_SWIZZLER{};

  // Note that for address spaces being swizzled (like O_sm, dO_sm/dO_sm_swap, and S_sm/S_sm_swap),
  // the start addresses of them need to be aligned with a transaction boundary
  // (i.e. 128-byte aligned), so that the designated swizzling configuration works correctly.
  //
  // Therefore, they are all allocated first to exploit the fact that
  // shared memory buffers are naturally 128-byte aligned, and
  // the size of each swizzled memery area is a multiple of the size of a transaction by design

  // (Br, v_d)
  Tensor dO_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head), SM_SWIZZLER), O_dO_sm_layout);
  moving_head += cosize(O_dO_sm_layout) * sizeof(T);

  // (Br, Bc)
  Tensor S_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head), SM_SWIZZLER), S_sm_layout);
  // Note that O_sm and S_sm share the same memory area, but occupy different column sizes
  // (therfore having different layouts), and the memory size reserved for them
  // should be the maximum of the two
  moving_head += max(cosize(S_sm_layout), cosize(O_dO_sm_layout)) * sizeof(T);

  // Since the size of L_T might be larger than T,
  // the space of l_sm is appropriated before others to avoid alignment errors

  // (Br, 1)
  Tensor l_sm = make_tensor(make_smem_ptr(reinterpret_cast<L_T*>(moving_head)), l_m_sm_layout);
  //Tensor l_sm_T = recast<T>(l_sm);
  moving_head += cosize(l_m_sm_layout) * sizeof(L_T);
#endif

  // (v_d, Br)
  Tensor dO_sm_swap = make_tensor(dO_sm.data(), select<1,0>(O_dO_sm_layout));

  // (Bc, Br)
  Tensor S_sm_swap = make_tensor(S_sm.data(), select<1,0>(S_sm_layout));

  // (Br, v_d)
  Tensor O_sm = make_tensor(S_sm.data(), O_dO_sm_layout);

  // (Br, d)
  Tensor Q_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), Q_sm_layout);
  // (d, Br)
  Tensor Q_sm_swap = make_tensor(Q_sm.data(), select<1,0>(Q_sm_layout));
  moving_head += cosize(Q_sm_layout) * sizeof(T);

  const auto K_dK_memory_size = cosize(K_dK_sm_layout) * sizeof(T);
  // (Bc, d)
  Tensor K_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), K_dK_sm_layout);
  // (d, Bc)
  Tensor K_sm_swap = make_tensor(K_sm.data(), select<1,0>(K_dK_sm_layout));
  moving_head += K_dK_memory_size;

  // (Bc, d)
  Tensor dK_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), K_dK_sm_layout);
  moving_head += K_dK_memory_size;

  const auto V_dV_memory_size = cosize(V_dV_sm_layout) * sizeof(T);
  // (Bc, v_d)
  Tensor V_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), V_dV_sm_layout);
  moving_head += V_dV_memory_size;

  Tensor dV_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), V_dV_sm_layout);
  moving_head += V_dV_memory_size;

  // (Br, Bc)
  Tensor P_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), P_sm_layout);
  // (Bc, Br)
  Tensor P_sm_swap = make_tensor(P_sm.data(), select<1,0>(P_sm_layout));
  moving_head += cosize(P_sm_layout) * sizeof(T);

  // (Br, 1)
  Tensor m_sm = make_tensor(make_smem_ptr(reinterpret_cast<T*>(moving_head)), l_m_sm_layout);
  moving_head += cosize(l_m_sm_layout) * sizeof(T);

  //
  // Tensors wrapping respective global memory pointers
  //

  // (q, d, b)
  const Tensor full_Q = make_tensor(make_gmem_ptr(Q), Q_dQ_layout);
  // (k, d, b)
  const Tensor full_K = make_tensor(make_gmem_ptr(K), K_dK_layout);
  // (k, v_d, b)
  const Tensor full_V = make_tensor(make_gmem_ptr(V), V_dV_layout);
  // (q, v_d, b)
  const Tensor full_O = make_tensor(make_gmem_ptr(O), O_dO_layout);
  // (q, v_d, b)
  const Tensor full_dO = make_tensor(make_gmem_ptr(dO), O_dO_layout);

  // (q, 1, b)
  const Tensor full_l = make_tensor(make_gmem_ptr(l), l_m_layout);
  const Tensor full_m = make_tensor(make_gmem_ptr(m), l_m_layout);

  // (q, d, b)
  Tensor full_dQ = make_tensor(make_gmem_ptr(dQ), Q_dQ_layout);
  // (k, d, b)
  Tensor full_dK = make_tensor(make_gmem_ptr(dK), K_dK_layout);
  // (k, v_d, b)
  Tensor full_dV = make_tensor(make_gmem_ptr(dV), V_dV_layout);

  //
  // Identity maps providing coordinates used by predicators for range/boundary checks
  //

  const Tensor full_Q_dQ_map = make_identity_tensor(full_Q.shape());
  const Tensor full_K_dK_map = make_identity_tensor(full_K.shape());
  const Tensor full_V_dV_map = make_identity_tensor(full_V.shape());
  const Tensor full_O_dO_map = make_identity_tensor(full_O.shape());
  const Tensor full_l_m_map = make_identity_tensor(full_l.shape());


  const Tensor Q_sm_map = make_identity_tensor(Q_sm.shape());
  const Tensor Q_sm_swap_map = make_identity_tensor(Q_sm_swap.shape());
  const Tensor K_dK_sm_map = make_identity_tensor(K_sm.shape());
  const Tensor K_sm_swap_map = make_identity_tensor(K_sm_swap.shape());
  const Tensor V_dV_sm_map = make_identity_tensor(V_sm.shape());
  const Tensor O_dO_sm_map = make_identity_tensor(O_sm.shape());
  const Tensor dO_sm_swap_map = make_identity_tensor(dO_sm_swap.shape());

  // P_sm/P_sm_swap and S_sm/S_sm_swap share the same map, as they have exactly the same shape
  const Tensor P_S_sm_map = make_identity_tensor(P_sm.shape());
  const Tensor P_S_sm_swap_map = make_identity_tensor(P_sm_swap.shape());

  // This map will be shared between l_sm and m_sm as they all have the same layout
  const Tensor l_m_sm_map = make_identity_tensor(l_sm.shape());

  //
  // Tiling tensors wrapping global memories;
  // blockIdx.x indexes select a Bc section,
  // while blockIdx.y selects a batch
  //

  const auto Br_rest_coords =  make_coord(_, Int<0>{}, blockIdx.y);
  const auto Bc_rest_coords = make_coord(blockIdx.x, Int<0>{}, blockIdx.y);

  const auto Q_O_rest_coords = make_coord(_, Int<0>{}, blockIdx.y);
  const auto K_V_rest_coords = make_coord(blockIdx.x, Int<0>{}, blockIdx.y);

  // (Br, d, # of Br sections)
  const Tensor Q_tile = local_tile(full_Q, Q_dQ_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);
  const Tensor Q_dQ_tile_map = local_tile(full_Q_dQ_map, Q_dQ_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);

  Tensor dQ_tile = local_tile(full_dQ, Q_dQ_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);


  // (Bc, d)
  const Tensor K_tile = local_tile(full_K, K_dK_tile_shape, Bc_rest_coords)(_, _, Int<0>{});
  const Tensor K_dK_tile_map = local_tile(full_K_dK_map, K_dK_tile_shape, Bc_rest_coords)(_, _, Int<0>{});

  Tensor dK_tile = local_tile(full_dK, K_dK_tile_shape, Bc_rest_coords)(_, _, Int<0>{});


  // (Bc, v_d)
  const Tensor V_tile = local_tile(full_V, V_dV_tile_shape, Bc_rest_coords)(_, _, Int<0>{});
  const Tensor V_dV_tile_map = local_tile(full_V_dV_map, V_dV_tile_shape, Bc_rest_coords)(_, _, Int<0>{});

  Tensor dV_tile = local_tile(full_dV, V_dV_tile_shape, Bc_rest_coords)(_, _, Int<0>{});

  // (Br, v_d, # of Br sections)
  const Tensor O_tile = local_tile(full_O, O_dO_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);
  const Tensor O_dO_tile_map = local_tile(full_O_dO_map, O_dO_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);

  const Tensor dO_tile = local_tile(full_dO, O_dO_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);


  // Br_occupancy is be set up to a tiled structure,
  // so it only takes a correct layout and indexing to access the target tile
  // (# of Br sections)
  Tensor Br_occupancy_tile = make_tensor(make_gmem_ptr(Br_occupancy),
                                    make_layout(make_shape(size<2>(O_tile), size<2>(full_O))))(_, blockIdx.y);

  // (Br, 1, # of Br sections)
  const Tensor l_tile = local_tile(full_l, l_m_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);
  const Tensor m_tile = local_tile(full_m, l_m_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);
  const Tensor l_m_tile_map = local_tile(full_l_m_map, l_m_tile_shape, Br_rest_coords)(_, _, Int<0>{}, _);





  //
  // Partitioning tensors with thread layouts
  //

  constexpr auto M0M_MODE0_THREAD_LAYOUT = select<0>(MODE0_MAJOR_THREAD_LAYOUT);
  constexpr auto M0M_MODE1_THREAD_LAYOUT = select<1>(MODE0_MAJOR_THREAD_LAYOUT);

  constexpr auto M1M_MODE0_THREAD_LAYOUT = select<0>(MODE1_MAJOR_THREAD_LAYOUT);
  constexpr auto M1M_MODE1_THREAD_LAYOUT = select<1>(MODE1_MAJOR_THREAD_LAYOUT);



  //
  // Q / dQ
  //

  // (# of thread groups along Br, # of thread groups along d, # of Br sections)
  const Tensor Q_tile_m0m_partitioned = PartitionPerThreadLayout(Q_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor Q_tile_map_m0m_partitioned = PartitionPerThreadLayout(Q_dQ_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, # of thread groups along d, # of Br sections)
  Tensor dQ_tile_m0m_partitioned = PartitionPerThreadLayout(dQ_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const auto& dQ_tile_map_m0m_partitioned = Q_tile_map_m0m_partitioned;

  // (# of thread groups along Br, # of thread groups along d)
  Tensor Q_sm_m0m_partitioned = PartitionPerThreadLayout(Q_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor Q_sm_map_m0m_partitioned = PartitionPerThreadLayout(Q_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, d)
  const Tensor Q_sm_m1m_mode0_partitioned = PartitionPerThreadLayout(Q_sm, M1M_MODE0_THREAD_LAYOUT);
  const Tensor Q_sm_map_m1m_mode0_partitioned = PartitionPerThreadLayout(Q_sm_map, M1M_MODE0_THREAD_LAYOUT);

  // (# of thread groups along d, Br)
  const Tensor Q_sm_swap_m0m_mode1_partitioned = PartitionPerThreadLayout(Q_sm_swap, M0M_MODE1_THREAD_LAYOUT);
  const Tensor Q_sm_swap_map_m0m_mode1_partitioned = PartitionPerThreadLayout(Q_sm_swap_map, M0M_MODE1_THREAD_LAYOUT);


  //
  // K / dK
  //

  // (# of thread groups along Bc, # of thread groups along d)
  const Tensor K_tile_m0m_partitioned = PartitionPerThreadLayout(K_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor K_tile_map_m0m_partitioned = PartitionPerThreadLayout(K_dK_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor K_sm_m0m_partitioned = PartitionPerThreadLayout(K_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor K_dK_sm_map_m0m_partitioned = PartitionPerThreadLayout(K_dK_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor dK_tile_m0m_partitioned = PartitionPerThreadLayout(dK_tile, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor dK_sm_m0m_partitioned = PartitionPerThreadLayout(dK_sm, MODE0_MAJOR_THREAD_LAYOUT);
  //const auto& dK_sm_map_m0m_partitioned = K_sm_map_m0m_partitioned;

  // (# of thread groups along Br, d)
  const Tensor K_sm_m1m_mode1_partitioned = PartitionPerThreadLayout(K_sm, M1M_MODE1_THREAD_LAYOUT);
  const Tensor K_sm_map_m1m_mode1_partitioned = PartitionPerThreadLayout(K_dK_sm_map, M1M_MODE1_THREAD_LAYOUT);

  // (# of thread groups along d, Bc)
  const Tensor K_sm_swap_m0m_mode1_partitioned = PartitionPerThreadLayout(K_sm_swap, M0M_MODE1_THREAD_LAYOUT);
  const Tensor K_sm_swap_map_m0m_mode1_partitioned = PartitionPerThreadLayout(K_sm_swap_map, M0M_MODE1_THREAD_LAYOUT);


  //
  // V / dV
  //

  // (# of thread groups along Bc, # of thread groups along v_d)
  const Tensor V_tile_m0m_partitioned = PartitionPerThreadLayout(V_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor V_tile_map_m0m_partitioned = PartitionPerThreadLayout(V_dV_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor V_sm_m0m_partitioned = PartitionPerThreadLayout(V_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor V_dV_sm_map_m0m_partitioned = PartitionPerThreadLayout(V_dV_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor dV_tile_m0m_partitioned = PartitionPerThreadLayout(dV_tile, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor dV_sm_m0m_partitioned = PartitionPerThreadLayout(dV_sm, MODE0_MAJOR_THREAD_LAYOUT);
  //const auto& dV_sm_map_m0m_partitioned = V_sm_map_m0m_partitioned;

  // (# of thread groups along Bc, v_d)
  const Tensor V_sm_m1m_mode1_partitioned = PartitionPerThreadLayout(V_sm, M1M_MODE1_THREAD_LAYOUT);
  const Tensor V_sm_map_m1m_mode1_partitioned = PartitionPerThreadLayout(V_dV_sm_map, M1M_MODE1_THREAD_LAYOUT);


  //
  // O / dO
  //

  // (# of thread groups along Br, # of thread groups along v_d, # of Br sections)
  const Tensor O_tile_m0m_partitioned = PartitionPerThreadLayout(O_tile, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor O_tile_map_m0m_partitioned = PartitionPerThreadLayout(O_dO_tile_map, MODE0_MAJOR_THREAD_LAYOUT);

  const Tensor dO_tile_m0m_partitioned = PartitionPerThreadLayout(dO_tile, MODE0_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Br, # of thread groups along v_d)
  Tensor O_sm_m0m_partitioned = PartitionPerThreadLayout(O_sm, MODE0_MAJOR_THREAD_LAYOUT);
  const Tensor O_sm_map_m0m_partitioned = PartitionPerThreadLayout(O_dO_sm_map, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor dO_sm_m0m_partitioned = PartitionPerThreadLayout(dO_sm, MODE0_MAJOR_THREAD_LAYOUT);

  Tensor O_sm_m1m_partitioned = PartitionPerThreadLayout(O_sm, MODE1_MAJOR_THREAD_LAYOUT);
  const Tensor O_sm_map_m1m_partitioned = PartitionPerThreadLayout(O_dO_sm_map, MODE1_MAJOR_THREAD_LAYOUT);
  const Tensor dO_sm_m1m_partitioned = PartitionPerThreadLayout(dO_sm, MODE1_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along v_d, Br)
  const Tensor dO_sm_swap_m0m_mode1_partitioned = PartitionPerThreadLayout(dO_sm_swap, M0M_MODE1_THREAD_LAYOUT);
  const Tensor dO_sm_swap_map_m0m_mode1_partitioned = PartitionPerThreadLayout(dO_sm_swap_map, M0M_MODE1_THREAD_LAYOUT);

  // (# of thread groups along Br, v_d)
  Tensor dO_sm_m1m_mode0_partitioned = PartitionPerThreadLayout(dO_sm, M1M_MODE0_THREAD_LAYOUT);
  const Tensor dO_sm_map_m1m_mode0_partitioned = PartitionPerThreadLayout(O_dO_sm_map, M1M_MODE0_THREAD_LAYOUT);


  //
  // l / m
  //
  // Note that the dimension 1 (indices start from 0) of l/m is just 1;
  // therefore that dimension can be indexed out directly

  // (# of thread groups along Br, # of Br sections)
  Tensor l_tile_m0m_partitioned_col0 = PartitionPerThreadLayout(l_tile, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);
  Tensor m_tile_m0m_partitioned_col0 = PartitionPerThreadLayout(m_tile, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);

  const Tensor l_m_tile_map_m0m_partitioned_col0 = PartitionPerThreadLayout(l_m_tile_map, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{}, _);


  // (# of thread groups along Br)
  Tensor l_sm_m0m_partitioned_col0 = PartitionPerThreadLayout(l_sm, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});
  Tensor m_sm_m0m_partitioned_col0 = PartitionPerThreadLayout(m_sm, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});

  const Tensor l_m_sm_map_m0m_partitioned_col0 = PartitionPerThreadLayout(l_m_sm_map, MODE0_MAJOR_THREAD_LAYOUT)(_, Int<0>{});


  const Tensor l_sm_m0m_mode0_partitioned_col0 = PartitionPerThreadLayout(l_sm, M0M_MODE0_THREAD_LAYOUT)(_, Int<0>{});

  const Tensor l_sm_m1m_mode0_partitioned_col0 = PartitionPerThreadLayout(l_sm, M1M_MODE0_THREAD_LAYOUT)(_, Int<0>{});
  // The memory of m_sm is reused to store D
  Tensor D_sm_m1m_mode0_partitioned_col0 = PartitionPerThreadLayout(m_sm, M1M_MODE0_THREAD_LAYOUT)(_, Int<0>{});
  const auto& m_sm_m1m_mode0_partitioned_col0 = D_sm_m1m_mode0_partitioned_col0;


  //
  // P
  //

  // (# of thread groups along Br, # of thread groups along Bc)
  Tensor P_sm_m1m_partitioned = PartitionPerThreadLayout(P_sm, MODE1_MAJOR_THREAD_LAYOUT);
  const Tensor P_sm_map_m1m_partitioned = PartitionPerThreadLayout(P_S_sm_map, MODE1_MAJOR_THREAD_LAYOUT);

  // (# of thread groups along Bc, # of thread groups along Br)
  const Tensor P_sm_swap_m0m_mode0_partitioned = PartitionPerThreadLayout(P_sm_swap, M0M_MODE0_THREAD_LAYOUT);
  const Tensor P_sm_swap_map_m0m_mode0_partitioned = PartitionPerThreadLayout(P_S_sm_swap_map, M0M_MODE0_THREAD_LAYOUT);


  //
  // S
  //

  // (# of thread groups along Br, # of thread groups along Bc)

  Tensor S_sm_m1m_partitioned = PartitionPerThreadLayout(S_sm, MODE1_MAJOR_THREAD_LAYOUT);
  const auto& S_sm_map_m1m_partitioned = P_sm_map_m1m_partitioned;


  // (# of thread groups along Bc, Br)
  const Tensor S_sm_swap_m0m_mode0_partitioned = PartitionPerThreadLayout(S_sm_swap, M0M_MODE0_THREAD_LAYOUT);
  const auto& S_sm_swap_map_m0m_mode0_partitioned = P_sm_swap_map_m0m_mode0_partitioned;

  // (# of thread groups along Br, Bc)
  const Tensor S_sm_m0m_mode0_partitioned = PartitionPerThreadLayout(S_sm, M0M_MODE0_THREAD_LAYOUT);
  const Tensor S_sm_map_m0m_mode0_partitioned = PartitionPerThreadLayout(P_S_sm_map, M0M_MODE0_THREAD_LAYOUT);


 //
 // boundary_check predicators which can be created in advance
 //

  const auto Q_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(Q_sm_map_m0m_partitioned, Q_sm.shape());
  const auto K_dK_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(K_dK_sm_map_m0m_partitioned, K_sm.shape());
  const auto V_dV_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(V_dV_sm_map_m0m_partitioned, V_sm.shape());
  const auto O_sm_map_m0m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(O_sm_map_m0m_partitioned, O_sm.shape());
  const auto O_sm_map_m1m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(O_sm_map_m1m_partitioned, O_sm.shape());


  const auto P_sm_map_m1m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(P_sm_map_m1m_partitioned, P_sm.shape());
  const auto S_sm_map_m1m_partitioned_boundary_check = GetBoundaryCheckPred<0, 1>(S_sm_map_m1m_partitioned, S_sm.shape());


  const auto l_m_sm_map_m0m_partitioned_col0_boundary_check = GetBoundaryCheckPred<0, 1>(l_m_sm_map_m0m_partitioned_col0, l_m_sm_layout.shape());

  #if 0
  //
  // alibi
  //

  const auto alibi_params = GetAlibiParameters(reference_seq_shape, log_alibi_slope_base);
  const T alibi_slope = get<0>(alibi_params);
  const auto alibi_cal_normaliser = static_cast<T>(get<1>(alibi_params));

  const auto normalised_diff_squared = [alibi_cal_normaliser](const auto& a, const auto& b) {
    const auto normalised = static_cast<decltype(alibi_cal_normaliser)>(a-b) / alibi_cal_normaliser;
    return normalised * normalised;
  };
  #endif

  //
  // Various functors/lambda functions
  //

  //static constexpr Copy_Atom<UniversalCopy<T>, T> copy_atom{};

  static constexpr UniversalCopy<T> copy_op{};
  static constexpr UniversalCopy<L_T> copy_op_L{};


  const T masking_value = TypeUtil<T>::GetNegInfApprox();

  static constexpr TrivialPredTensor always_trues{};


  static constexpr plus addition_func{};
  static constexpr multiplies multiplication_func{};



  const auto dS_calc_func = [&dot_scaler] (const T& dP, const T&, const T& P, const T& D) constexpr {
                                            return P * (dP - D) * dot_scaler;
                                          };

  static constexpr auto addition_post_func = [](const T& current_value, const auto& existing_value) constexpr {
                                                      return current_value + const_cast<const T&>(existing_value);
                                                    };

  //
  // Additional register block partitioning for GEMM0/GEMM1
  //

  // Block sizes of gemm "O = M @ N" below are chosen based on the thread layout used
  // and my test results, so they might not be optimal under different threading configurations
  // and/or with GPUs of different capabilities
  static constexpr auto GEMM0_M_REGISTER_BLOCK_TILER = make_shape(Int<1>{}, Int<4>{});
  static constexpr auto GEMM0_N_REGISTER_BLOCK_TILER = make_shape(Int<2>{}, Int<4>{});
  static constexpr auto GEMM0_O_REGISTER_BLOCK_TILER = make_shape(get<0>(GEMM0_M_REGISTER_BLOCK_TILER), get<0>(GEMM0_N_REGISTER_BLOCK_TILER));

  static constexpr auto GEMM1_M_REGISTER_BLOCK_TILER = make_shape(Int<2>{}, Int<4>{});
  static constexpr auto GEMM1_N_REGISTER_BLOCK_TILER = make_shape(Int<1>{}, Int<4>{});
  static constexpr auto GEMM1_O_REGISTER_BLOCK_TILER = make_shape(get<0>(GEMM1_M_REGISTER_BLOCK_TILER), get<0>(GEMM1_N_REGISTER_BLOCK_TILER));



  // (GEMM1_M_REGISTER_BLOCK_SHAPE, (rest dimension associated with Br, rest dimension associated with d))
  const Tensor Q_sm_m1m_mode0_partitioned_gemm_block = zipped_divide(Q_sm_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);
  const Tensor Q_sm_map_m1m_mode0_partitioned_gemm_block = zipped_divide(Q_sm_map_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);


  // (GEMM1_M_REGISTER_BLOCK_SHAPE, (rest dimension associated with Br, rest dimension associated with v_d))
  const Tensor dO_sm_m1m_mode0_partitioned_gemm_block = zipped_divide(dO_sm_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);
  const Tensor dO_sm_map_m1m_mode0_partitioned_gemm_block = zipped_divide(dO_sm_map_m1m_mode0_partitioned, GEMM1_M_REGISTER_BLOCK_TILER);


  // (GEMM0_M_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with Bc))
  const Tensor S_sm_m0m_mode0_partitioned_gemm_block = zipped_divide(S_sm_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);
  const Tensor S_sm_map_m0m_mode0_partitioned_gemm_block = zipped_divide(S_sm_map_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);


  // (GEMM0_M_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with Br))
  const Tensor P_sm_swap_m0m_mode0_partitioned_gemm_block = zipped_divide(P_sm_swap_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);
  const Tensor P_sm_swap_map_m0m_mode0_partitioned_gemm_block = zipped_divide(P_sm_swap_map_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);

  const Tensor S_sm_swap_m0m_mode0_partitioned_gemm_block = zipped_divide(S_sm_swap_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);
  const Tensor S_sm_swap_map_m0m_mode0_partitioned_gemm_block = zipped_divide(S_sm_swap_map_m0m_mode0_partitioned, GEMM0_M_REGISTER_BLOCK_TILER);


  // (GEMM1_N_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with d))
  const Tensor K_sm_m1m_mode1_partitioned_gemm_block = zipped_divide(K_sm_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);
  const Tensor K_sm_map_m1m_mode1_partitioned_gemm_block = zipped_divide(K_sm_map_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);


  // (GEMM1_N_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with v_d))
  const Tensor V_sm_m1m_mode1_partitioned_gemm_block = zipped_divide(V_sm_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);
  const Tensor V_sm_map_m1m_mode1_partitioned_gemm_block = zipped_divide(V_sm_map_m1m_mode1_partitioned, GEMM1_N_REGISTER_BLOCK_TILER);


  // (GEMM0_N_REGISTER_BLOCK_TILER, (rest dimension associated with d, rest dimension associated with Bc))
  const Tensor K_sm_swap_m0m_mode1_partitioned_gemm_block = zipped_divide(K_sm_swap_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);
  const Tensor K_sm_swap_map_m0m_mode1_partitioned_gemm_block = zipped_divide(K_sm_swap_map_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);


  // (GEMM0_N_REGISTER_BLOCK_TILER, (rest dimension associated with v_d, rest dimension associated with Br))
  const Tensor dO_sm_swap_m0m_mode1_partitioned_gemm_block = zipped_divide(dO_sm_swap_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);
  const Tensor dO_sm_swap_map_m0m_mode1_partitioned_gemm_block = zipped_divide(dO_sm_swap_map_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);


  // (GEMM0_N_REGISTER_BLOCK_TILER, (rest dimension associated with d, rest dimension associated with Br))
  const Tensor Q_sm_swap_m0m_mode1_partitioned_gemm_block = zipped_divide(Q_sm_swap_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);
  const Tensor Q_sm_swap_map_m0m_mode1_partitioned_gemm_block = zipped_divide(Q_sm_swap_map_m0m_mode1_partitioned, GEMM0_N_REGISTER_BLOCK_TILER);


  // (GEMM1_O_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with Bc))
  Tensor P_sm_m1m_partitioned_gemm_block = zipped_divide(P_sm_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);
  const Tensor P_sm_map_m1m_partitioned_gemm_block = zipped_divide(P_sm_map_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);

  const auto GEMM1_O_REGISTER_BLOCK_TILER_MODE0 = select<0>(GEMM1_O_REGISTER_BLOCK_TILER);
  // (GEMM1_O_REGISTER_BLOCK_TILER[0:1], (rest dimension associated with Br))
  const Tensor l_sm_m1m_mode0_partitioned_col0_gemm_block = zipped_divide(l_sm_m1m_mode0_partitioned_col0, GEMM1_O_REGISTER_BLOCK_TILER_MODE0);
  const Tensor m_sm_m1m_mode0_partitioned_col0_gemm_block = zipped_divide(m_sm_m1m_mode0_partitioned_col0, GEMM1_O_REGISTER_BLOCK_TILER_MODE0);
  const auto& D_sm_m1m_mode0_partitioned_col0_gemm_block = m_sm_m1m_mode0_partitioned_col0_gemm_block;

  // (GEMM1_O_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with Bc))
  Tensor S_sm_m1m_partitioned_gemm_block = zipped_divide(S_sm_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);
  const Tensor S_sm_map_m1m_partitioned_gemm_block = zipped_divide(S_sm_map_m1m_partitioned, GEMM1_O_REGISTER_BLOCK_TILER);

  // (GEMM0_O_REGISTER_BLOCK_TILER, (rest dimension associated with Br, rest dimension associated with d), # of Br sections)
  const Tensor dQ_tile_m0m_partitioned_gemm_block = group_modes<1,3>(zipped_divide(dQ_tile_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER)(_, repeat<3>(_)));
  const Tensor dQ_tile_map_m0m_partitioned_gemm_block = group_modes<1,3>(zipped_divide(dQ_tile_map_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER)(_, repeat<3>(_)));


  // (GEMM0_O_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with v_d)))
  const Tensor dV_sm_m0m_partitioned_gemm_block = zipped_divide(dV_sm_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER);
  const Tensor dV_sm_map_m0m_partitioned_gemm_block = zipped_divide(V_dV_sm_map_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER);


  // (GEMM0_O_REGISTER_BLOCK_TILER, (rest dimension associated with Bc, rest dimension associated with d)))
  const Tensor dK_sm_m0m_partitioned_gemm_block = zipped_divide(dK_sm_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER);
  const Tensor dK_sm_map_m0m_partitioned_gemm_block = zipped_divide(K_dK_sm_map_m0m_partitioned, GEMM0_O_REGISTER_BLOCK_TILER);


  const auto Q_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(Q_sm_map_m1m_mode0_partitioned_gemm_block, Q_sm.shape());
  const auto dO_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(dO_sm_map_m1m_mode0_partitioned_gemm_block, dO_sm.shape());
  const auto S_sm_map_m0m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(S_sm_map_m0m_mode0_partitioned_gemm_block, S_sm.shape());
  const auto P_sm_swap_map_m0m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(P_sm_swap_map_m0m_mode0_partitioned_gemm_block, P_sm_swap.shape());
  const auto S_sm_swap_map_m0m_mode0_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(S_sm_swap_map_m0m_mode0_partitioned_gemm_block, S_sm_swap.shape());


  const auto K_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(K_sm_map_m1m_mode1_partitioned_gemm_block, K_sm.shape());
  const auto V_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(V_sm_map_m1m_mode1_partitioned_gemm_block, V_sm.shape());
  const auto K_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(K_sm_swap_map_m0m_mode1_partitioned_gemm_block, K_sm_swap.shape());
  const auto dO_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(dO_sm_swap_map_m0m_mode1_partitioned_gemm_block, dO_sm_swap.shape());
  const auto Q_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(Q_sm_swap_map_m0m_mode1_partitioned_gemm_block, Q_sm_swap.shape());


  const auto P_sm_map_m1m_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(P_sm_map_m1m_partitioned_gemm_block, P_sm.shape());
  const auto S_sm_map_m1m_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(S_sm_map_m1m_partitioned_gemm_block, S_sm.shape());
  const auto dV_sm_map_m0m_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(dV_sm_map_m0m_partitioned_gemm_block, dV_sm.shape());
  const auto dK_sm_map_m0m_partitioned_gemm_block_boundary_check = GetBoundaryCheckPred<0, 1>(dK_sm_map_m0m_partitioned_gemm_block, dK_sm.shape());


  //
  // Loading data and performing forward calculation
  //


  const auto first_K_index = blockIdx.x * get<0>(K_dK_tile_shape);
  const auto last_K_index = (blockIdx.x+1) * get<0>(K_dK_tile_shape) - 1;
  const auto bbox_min_for_K = make_tuple(min(first_K_index, get<0>(full_K.shape())-1), Int<0>{}, blockIdx.y);
  const auto bounded_last_K_index = min(first_K_index+get<0>(K_dK_tile_shape)-1, get<0>(full_K.shape())-1);

  const auto max_order = size(reference_seq_shape) - 1;
  const auto min_k_order = get<0>(K_seq_order_map(first_K_index));
  const auto max_k_order = min(get<0>(K_seq_order_map(last_K_index)), max_order);

  // Loading the target block of K and V into K_sm and V_sm respectively and clearing dK_sm and dV_sm,
  // in one single loop rather than multiple loops for each loading/clearing
  const auto K_tile_map_m0m_partitioned_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(K_tile_map_m0m_partitioned,
                            bbox_min_for_K,
                            make_tuple(bounded_last_K_index, get<1>(full_K.shape())-1, blockIdx.y));
  const auto& dK_tile_map_m0m_partitioned_boundary_check = K_tile_map_m0m_partitioned_boundary_check;


  const auto V_tile_map_m0m_partitioned_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(V_tile_map_m0m_partitioned,
                              bbox_min_for_K,
                              make_tuple(bounded_last_K_index, get<1>(full_V.shape())-1, blockIdx.y));
  const auto& dV_tile_map_m0m_partitioned_boundary_check = V_tile_map_m0m_partitioned_boundary_check;

  const auto max_size_of_K_V_sm_m0m_partitioning = max(size(K_sm_m0m_partitioned), size(V_sm_m0m_partitioned));

  CUTE_UNROLL
  for (int scanning_index = 0; scanning_index < max_size_of_K_V_sm_m0m_partitioning; ++scanning_index) {
    if (K_dK_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
      if (K_tile_map_m0m_partitioned_boundary_check(scanning_index))
        copy_op.copy(K_tile_m0m_partitioned(scanning_index), K_sm_m0m_partitioned(scanning_index));
      else
        K_sm_m0m_partitioned(scanning_index) = T(0);

      dK_sm_m0m_partitioned(scanning_index) = T(0);
    }

    if (V_dV_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
      if (V_tile_map_m0m_partitioned_boundary_check(scanning_index))
        copy_op.copy(V_tile_m0m_partitioned(scanning_index), V_sm_m0m_partitioned(scanning_index));
      else
        V_sm_m0m_partitioned(scanning_index) = T(0);

      dV_sm_m0m_partitioned(scanning_index) = T(0);
    }
  }

  // Having loops over Br sections processed by each thread block start at different indices,
  // to decrease the possiblity of racing for updating the same data sections assoicated with
  // the same Br index
  const int num_of_Br = size(Br_occupancy_tile);
  const int Br_index_start = blockIdx.x * max(1, num_of_Br / gridDim.x);

  CUTE_UNROLL
  for (int Br_scan = 0; Br_scan < num_of_Br; ++Br_scan) {
    int Br_index = (Br_index_start + Br_scan) % num_of_Br;

    const auto first_Q_index = Br_index * get<0>(Q_dQ_tile_shape);
    const auto last_Q_index = (Br_index+1) * get<0>(Q_dQ_tile_shape) - 1;

    // Early Checking to see if we can skip this iteration entirely.
    // For example, when doing causal attention, no need to process blocks of Q in which all examples are located before the current block of K
    if (attention_policy.IsSkipped(reference_seq_shape,
                                    get<0>(Q_seq_order_map(first_Q_index)), min(get<0>(Q_seq_order_map(last_Q_index)), max_order),
                                    min_k_order, max_k_order)) {
      continue;
    }

    const auto bbox_min_for_Q = make_tuple(min(first_Q_index, get<0>(full_Q.shape())-1), Int<0>{}, blockIdx.y);
    const auto bounded_last_Q_index = min(last_Q_index, get<0>(full_Q.shape())-1);


    // Since Q, O, dO, l, and m are globally constant during the whole process,
    // it is fine to load data from them for this run of loops first,
    // then wait for the Br_occupancy_indicator to be cleared,
    // so that only one __syncthreads() is needed for these operations

    const auto Q_tile_map_m0m_partitioned_Br_indexed_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(Q_tile_map_m0m_partitioned(_, _, Br_index),
                                          bbox_min_for_Q,
                                          make_tuple(bounded_last_Q_index, get<1>(full_Q.shape())-1, blockIdx.y));


    const auto O_tile_map_m0m_partitioned_Br_indexed_boundary_check =
              GetBoundaryCheckPred<0, 1, 2>(O_tile_map_m0m_partitioned(_, _, Br_index),
                                          bbox_min_for_Q,
                                          make_tuple(bounded_last_Q_index, get<1>(full_dO.shape())-1, blockIdx.y));

    const auto Q_tile_m0m_partitioned_Br_indexed = Q_tile_m0m_partitioned(_, _, Br_index);
    const auto O_tile_m0m_partitioned_Br_indexed = O_tile_m0m_partitioned(_, _, Br_index);
    const auto dO_tile_m0m_partitioned_Br_indexed = dO_tile_m0m_partitioned(_, _, Br_index);

    const auto max_size_of_Q_O_sm_m0m_partitioning = max(size(Q_sm_m0m_partitioned), size(O_sm_m0m_partitioned));

    // Loading Q, O, dO in one single loop,
    CUTE_UNROLL
    for (int scanning_index = 0; scanning_index < max_size_of_Q_O_sm_m0m_partitioning; ++scanning_index) {
      if (Q_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
        if (Q_tile_map_m0m_partitioned_Br_indexed_boundary_check(scanning_index))
          copy_op.copy(Q_tile_m0m_partitioned_Br_indexed(scanning_index), Q_sm_m0m_partitioned(scanning_index));
        else
          Q_sm_m0m_partitioned(scanning_index) = T(0);

      }

      if (O_sm_map_m0m_partitioned_boundary_check(scanning_index)) {
        if (O_tile_map_m0m_partitioned_Br_indexed_boundary_check(scanning_index)) {
          copy_op.copy(O_tile_m0m_partitioned_Br_indexed(scanning_index), O_sm_m0m_partitioned(scanning_index));
          copy_op.copy(dO_tile_m0m_partitioned_Br_indexed(scanning_index), dO_sm_m0m_partitioned(scanning_index));
        }
        else {
          O_sm_m0m_partitioned(scanning_index) = T(0);
          dO_sm_m0m_partitioned(scanning_index) = T(0);
        }
      }
    }
    // Loading l and m in a separate loop,
    // as it takes much fewer iterations
    const auto l_m_tile_map_m0m_partitioned_col0_Br_indexed_boundary_check
                = GetBoundaryCheckPred<0, 1, 2>(l_m_tile_map_m0m_partitioned_col0(_, Br_index),
                                  bbox_min_for_Q,
                                  make_tuple(bounded_last_Q_index, get<1>(full_l.shape())-1, blockIdx.y));

    const auto l_tile_m0m_partitioned_col0_Br_indexed = l_tile_m0m_partitioned_col0(_, Br_index);
    const auto m_tile_m0m_partitioned_col0_Br_indexed = m_tile_m0m_partitioned_col0(_, Br_index);

    CUTE_UNROLL
    for (int scanning_index = 0; scanning_index < size(l_sm_m0m_partitioned_col0); ++scanning_index) {
      if (l_m_sm_map_m0m_partitioned_col0_boundary_check(scanning_index)) {
        if (l_m_tile_map_m0m_partitioned_col0_Br_indexed_boundary_check(scanning_index)) {
          copy_op_L.copy(l_tile_m0m_partitioned_col0_Br_indexed(scanning_index), l_sm_m0m_partitioned_col0(scanning_index));
          copy_op.copy(m_tile_m0m_partitioned_col0_Br_indexed(scanning_index), m_sm_m0m_partitioned_col0(scanning_index));
        }
        else {
          l_sm_m0m_partitioned_col0(scanning_index) = L_T(0);
          m_sm_m0m_partitioned_col0(scanning_index) = masking_value;
        }
      }
    }


    // Only thread 0 checks if this block can proceed to process this Br section,
    // while others are waiting at the __syncthreads() below
    if (threadIdx.x == 0) {
      // Note that BrOccupancyValueType does not contain volatile qualifier
      using BrOccupancyValueType = typename decltype(Br_occupancy_tile)::value_type;
      BrOccupancyValueType* Br_occupancy_indicator = const_cast<BrOccupancyValueType*>(&Br_occupancy_tile(Br_index));
      while (atomicCAS(Br_occupancy_indicator, 0, 1)) ;
    }

    __syncthreads();



    const auto p_calc_func = [Br_index,
                              &P_sm_map_m1m_partitioned_gemm_block,
                              &Q_dQ_tile_map, &Q_dQ_layout,
                              &K_dK_tile_map, &K_dK_layout,
                              &reference_seq_shape, &Q_seq_order_map, &K_seq_order_map, &attention_policy,
                              dot_scaler
                              ] (const T& current_value, const T&, const auto& coords, const T& max_value, const L_T& normaliser) {
      const auto q = get<0>(Q_dQ_tile_map(get<0>(coords), Int<0>{}, Br_index));
      const auto q_order = get<0>(Q_seq_order_map(q));
      const auto k = get<0>(K_dK_tile_map(get<1>(coords), Int<0>{}));
      const auto k_order = get<0>(K_seq_order_map(k));

      if (q < size<0>(Q_dQ_layout) && k < size<0>(K_dK_layout)
              && attention_policy.Check(reference_seq_shape, q_order, k_order)) {

        T logit = current_value * dot_scaler;
        return static_cast<T>(static_cast<L_T>(TypeUtil<T>::Exp(logit - max_value)) / normaliser);

      } else {
        return T(0);
      }
    };



    // Computing P
    CopyToRegisterAndGEMM(
      copy_op,
      Q_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check,
      Q_sm_m1m_mode0_partitioned_gemm_block,
      K_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check,
      K_sm_m1m_mode1_partitioned_gemm_block,
      P_sm_map_m1m_partitioned_gemm_block_boundary_check,
      P_sm_m1m_partitioned_gemm_block,
      p_calc_func,
      P_sm_map_m1m_partitioned_gemm_block,
      m_sm_m1m_mode0_partitioned_col0_gemm_block,
      l_sm_m1m_mode0_partitioned_col0_gemm_block
    );


    __syncthreads();

    // Computing dV
    CopyToRegisterAndGEMM(
      copy_op,
      P_sm_swap_map_m0m_mode0_partitioned_gemm_block_boundary_check,
      P_sm_swap_m0m_mode0_partitioned_gemm_block,
      dO_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check,
      dO_sm_swap_m0m_mode1_partitioned_gemm_block,
      dV_sm_map_m0m_partitioned_gemm_block_boundary_check,
      dV_sm_m0m_partitioned_gemm_block,
      addition_post_func
    );


    // Computing D
    ReduceAlongMode1WithWarp(MODE1_MAJOR_THREAD_LAYOUT,
                      always_trues,
                      O_sm_map_m1m_partitioned_boundary_check,
                      dO_sm_m1m_partitioned,
                      D_sm_m1m_mode0_partitioned_col0,
                      0,
                      addition_func,
                      multiplication_func,
                      O_sm_m1m_partitioned
                      );


    __syncwarp();



    // Computing S
    CopyToRegisterAndGEMM(
      copy_op,
      dO_sm_map_m1m_mode0_partitioned_gemm_block_boundary_check,
      dO_sm_m1m_mode0_partitioned_gemm_block,
      V_sm_map_m1m_mode1_partitioned_gemm_block_boundary_check,
      V_sm_m1m_mode1_partitioned_gemm_block,
      S_sm_map_m1m_partitioned_gemm_block_boundary_check,
      S_sm_m1m_partitioned_gemm_block,
      dS_calc_func,
      P_sm_m1m_partitioned_gemm_block,
      D_sm_m1m_mode0_partitioned_col0_gemm_block
    );

    __syncthreads();



    // Updating dQ and writing back the result directly to its global memory
    CopyToRegisterAndGEMM(
      copy_op,
      S_sm_map_m0m_mode0_partitioned_gemm_block_boundary_check,
      S_sm_m0m_mode0_partitioned_gemm_block,
      K_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check,
      K_sm_swap_m0m_mode1_partitioned_gemm_block,
      GetBoundaryCheckPred<0, 1, 2>(dQ_tile_map_m0m_partitioned_gemm_block(_, _, Br_index),
                                          bbox_min_for_Q,
                                          make_tuple(bounded_last_Q_index, get<1>(full_dQ.shape())-1, blockIdx.y)),
      dQ_tile_m0m_partitioned_gemm_block(_, _, Br_index),
      addition_post_func
    );

    // Computing dK
    CopyToRegisterAndGEMM(
      copy_op,
      S_sm_swap_map_m0m_mode0_partitioned_gemm_block_boundary_check,
      S_sm_swap_m0m_mode0_partitioned_gemm_block,
      Q_sm_swap_map_m0m_mode1_partitioned_gemm_block_boundary_check,
      Q_sm_swap_m0m_mode1_partitioned_gemm_block,
      dK_sm_map_m0m_partitioned_gemm_block_boundary_check,
      dK_sm_m0m_partitioned_gemm_block,
      addition_post_func
    );

    // __threadfence() flushes all writes above back to L2/global memory,
    // and the following __syncthreads() makes sure that flushes triggered by all threads are done
    // before thread 0 goes on to reset the occupancy flag
    __threadfence();
    __syncthreads();
    // only thread 0 does the reset of the occupancy flag of this Br section while others go ahead
    if (threadIdx.x == 0) {
      Br_occupancy_tile(Br_index) = 0;
    }


  }

  // Writing back dK/dV to their respective global memories,
  // in one single loop rather than multiple loops for each write

  CUTE_UNROLL
  for (int scanning_index = 0; scanning_index < max_size_of_K_V_sm_m0m_partitioning; ++scanning_index) {
    if (dK_tile_map_m0m_partitioned_boundary_check(scanning_index))
      copy_op.copy(dK_sm_m0m_partitioned(scanning_index), dK_tile_m0m_partitioned(scanning_index));

    if (dV_tile_map_m0m_partitioned_boundary_check(scanning_index))
      copy_op.copy(dV_sm_m0m_partitioned(scanning_index), dV_tile_m0m_partitioned(scanning_index));
  }

}

} // namespace cuda_device_functions

namespace cuda_launch {

using namespace cuda_device_functions;
using namespace cute;


template <typename T>
static constexpr inline auto DecideConfigurationForForward(
        const int32_t shared_memory_available,
        const int32_t b,
        const int32_t q, const int32_t k,
        const int32_t d, const int32_t v_d) {

  // Explicitly declaring size variables as int32_t
  constexpr int32_t size_of_T = sizeof(T);

  static_assert(size_of_T % 2 == 0 && size_of_T >= 2, "sizeof(T) should be even and >= 2");

  // Br is set to a predefined number based on the size of T
  const auto Br = KernelConfig<T>::BR_SIZE;

  // This accounts for the memory requirement of l_sm, m_sm, and Q_sm
  const auto Br_associated_size = (1 + 1 + d) * size_of_T;

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  // This accounts for the memory requirement of K_sm, V_sm, and P_S_sm padded along the Br dimension;
  const auto Bc_associated_size = (d + v_d +
                                      (Br + KernelConfig<T>::PADDING_SIZE)
                                      ) * size_of_T;
#else
  // This accounts for the memory requirement of K_sm, V_sm, and P_S_sm
  const auto Bc_associated_size = (d + v_d + Br) * size_of_T;
#endif

  const auto available_Bc_memory = shared_memory_available - Br * Br_associated_size;

  const auto Bc = available_Bc_memory / Bc_associated_size;

  assert(Bc > 0 && "Failed to come up with a valid Bc");

  return make_tuple(Br, Bc);
}

template <typename T, typename L_T>
static constexpr inline auto DetermineConfigurationForBackward(
        const int32_t shared_memory_available,
        const int32_t b,
        const int32_t q, const int32_t k,
        const int32_t d, const int32_t v_d) {

  // Explicitly declaring size variables as int32_t
  constexpr int32_t size_of_T = sizeof(T), size_of_L_T = sizeof(L_T);

  static_assert(sizeof(T) % 2 == 0 && sizeof(T) >= 2, "sizeof(T) should be even and >= 2");

  // Br is set to a predefined number based on the size of T
  const auto Br = KernelConfig<T>::BR_SIZE;

  // This accounts for the memory requirement of Q_sm, dO_sm, m_sm, and l_sm
  const auto Br_associated_size = (d + v_d + 1) * size_of_T + 1 * size_of_L_T;

  // This accounts for the memory requirement of K_sm, dK_sm, V_sm, dV_sm, P_sm,
  const auto Bc_associated_size = (d*2 + v_d*2 + Br) * size_of_T;

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  constexpr auto PADDING_SIZE = KernelConfig<T>::PADDING_SIZE;

  // This accounts for the memory requirement of S_sm padded along the Br dimension specifically
  const auto Bc_associated_size_for_S_sm = (Br + PADDING_SIZE) * size_of_T;

  // This accounts for the padding size of dO_sm
  const auto Br_reservation_size = PADDING_SIZE * v_d * size_of_T;

  auto available_Bc_memory = shared_memory_available - Br * Br_associated_size - Br_reservation_size;
#else
  // This accounts for the memory requirement of S_sm specifically
  const auto Bc_associated_size_for_S_sm = Br * size_of_T;

  auto available_Bc_memory = shared_memory_available - Br * Br_associated_size;
#endif

  auto Bc = available_Bc_memory / (Bc_associated_size + Bc_associated_size_for_S_sm);

  // for the purpose of sharing memory space,
  // S_sm needs to be larger enough to cover the memory requirement of O_sm, i.e. Bc should be >= v_d;
  // so when Bc < v_d, which means S_sm fails to meet that requirement,
  // trying to apppropriate the required size directly and recalculating a new Bc
  if (Bc < v_d) {
    available_Bc_memory -= Bc_associated_size_for_S_sm * v_d;
    Bc = available_Bc_memory / Bc_associated_size;
  }

  assert(Bc > 0 && "Failed to come up with a valid Bc");

  return make_tuple(Br, Bc);

}

template <typename T, typename ReferenceSeqShape, typename SeqOrderMap, typename AttentionPolicy>
void FlashAttentionLauncher<T, ReferenceSeqShape, SeqOrderMap, AttentionPolicy>::EstimateForwardFlops(
        const SharedMemoryDescriptor& shared_memory_dest,
        const int32_t b,
        const int32_t q, const int32_t k,
        const int32_t d, const int32_t v_d,
        const ReferenceSeqShape& reference_seq_shape,
        const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
        const AttentionPolicy& attention_policy,
        float& flops) const {

  const auto [Br, Bc] = DecideConfigurationForForward<T>(shared_memory_dest.GetAvailableAmount(), b, q, k, d, v_d);
  const auto num_of_Br = (q + Br - 1) / Br;
  const auto num_of_Bc = (k + Bc - 1) / Bc;

  // Below are estimates of flops taken by various ops
  // on processing each pair of Bc and Br block
  //
  // Note that for the purpse of flops estimation, only primitive ops like "+", "-", "*", "/" are considered

  // flops for Q @ K^T
  const auto block_gemm_Q_K_flops = Br * Bc * (2*d - 1);

  // numerator_flops includes 2 reduction ops (max and sum along Bc)
  // and 3 elementwise ops (subtracting the max, exp(), and scaling)
  const auto numerator_flops = Br * (Bc - 1) * 2 + Br * Bc * 2;

  // Flops for ops updating l and m, as well as preparing the weights for O and P_S,
  //
  // m_new = max(m_current, m_tilde)
  // O_weight = exp(m_current - m_new) * l_current
  // P_S_weight = exp(m_tilde - m_new)
  // l_new = P_S_weight * l_tilde + O_weight
  // O_weight /= l_new
  // P_S_weight /= l_new
  //
  const auto l_m_update_flops = Br * 7;

  // reweighing O and P_S elementwise with O_weight and P_S_weight, respectively
  const auto reweighing_O_P_S_flops = Br * (Bc + v_d);

  // flops for P @ V
  const auto block_gemm_P_V_flops = Br * v_d * (2*Bc - 1);

  const auto total_flops_per_block_pair = block_gemm_Q_K_flops + numerator_flops + l_m_update_flops + reweighing_O_P_S_flops + block_gemm_P_V_flops;

  const auto max_order = size(reference_seq_shape) - 1;

  flops = 0.0;
  for (remove_cv_t<decltype(num_of_Bc)> Bc_index = 0; Bc_index < num_of_Bc; ++Bc_index) {
    const auto first_K_index = Bc_index * Bc;
    const auto last_K_index = (Bc_index+1) * Bc - 1;
    const auto min_k_order = get<0>(K_seq_order_map(first_K_index));
    const auto max_k_order = min(get<0>(K_seq_order_map(last_K_index)), max_order);

    for (remove_cv_t<decltype(num_of_Br)> Br_index = 0; Br_index < num_of_Br; ++Br_index) {

      const auto first_Q_index = Br_index * Br;
      const auto last_Q_index = (Br_index+1) * Br - 1;

      // The same check of IsSkipped() as in the forward kernel is verified here,
      // and when the result is true, no flops are inflicted as the computation
      // involving this pair of blocks indexed by (Bc_index, Br_index) will be
      // skipped in the actual forward process.
      if (attention_policy.IsSkipped(reference_seq_shape,
                                      get<0>(Q_seq_order_map(first_Q_index)), min(get<0>(Q_seq_order_map(last_Q_index)), max_order),
                                      min_k_order, max_k_order)) {
        continue;
      }
      else {
        flops += total_flops_per_block_pair;
      }
    }
  }

}


template <typename T, typename ReferenceSeqShape, typename SeqOrderMap, typename AttentionPolicy>
cudaError_t FlashAttentionLauncher<T, ReferenceSeqShape, SeqOrderMap, AttentionPolicy>::Forward(
        const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
        const int32_t b,
        const int32_t q, const int32_t k,
        const int32_t d, const int32_t v_d,
        const T* Q, const T* K, const T* V,
        T* O, L_T* l, T* m,
        uint32_t* Br_occupancy,
        const ReferenceSeqShape& reference_seq_shape,
        const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
        const AttentionPolicy& attention_policy) const {

  const auto [Br, Bc] = DecideConfigurationForForward<T>(shared_memory_dest.GetAvailableAmount(), b, q, k, d, v_d);

  T dot_scaler = TypeUtil<T>::RSqrt(static_cast<T>(d));

  #if 0
  T log_alibi_slope_base = TypeUtil<T>::GetNegInfApprox();
  if (alibi_on) {
    const T number_of_heads = static_cast<T>(get<decltype(rank(reference_seq_shape))::value-1>(reference_seq_shape));
    log_alibi_slope_base = TypeUtil<T>::Max(T(-8) / number_of_heads, T(-1));
  }
  #endif

  // The size of the x dimension of the thread grid
  // is set to the number of Bc sections needed to cover k,
  const auto grid_dim_x = (k + Bc - 1) / Bc;
  // The y dimension of the thread grid is set to b (the batch size)
  const auto grid_dim_y = b;

  //
  // Layouts for the global memories
  //

  const auto Q_layout = make_layout(make_shape(q, d, b));
  const auto K_layout = make_layout(make_shape(k, d, b));
  const auto V_layout = make_layout(make_shape(k, v_d, b));
  const auto O_layout = make_layout(make_shape(q, v_d, b));
  const auto l_m_layout = make_layout(make_shape(q, Int<1>{}, b));


  //
  // Tile shapes
  //

  // The batch dimension (mode 2, indices starting from 0) is not tiled,
  // and it will be indexed by the y coord of the thread grid
  const auto Q_tile_shape = make_shape(Br, d, Int<1>{});
  const auto K_tile_shape = make_shape(Bc, d, Int<1>{});
  const auto V_tile_shape = make_shape(Bc, v_d, Int<1>{});
  const auto O_tile_shape = make_shape(Br, v_d, Int<1>{});
  const auto l_m_tile_shape = make_shape(Br, Int<1>{}, Int<1>{});

  //
  // Layouts for the shared memories
  //

  const auto Q_sm_layout = make_layout(select<0, 1>(Q_tile_shape));
  const auto K_sm_layout = make_layout(select<0, 1>(K_tile_shape));
  const auto V_sm_layout = make_layout(select<0, 1>(V_tile_shape));
  const auto l_m_sm_layout = make_layout(select<0, 1>(l_m_tile_shape));

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  // S_P_sm is padded along the Br dimension by PADDING_SIZE
  const auto S_P_sm_layout = make_layout(make_shape(get<0>(Q_tile_shape), get<0>(K_tile_shape)),
                                            make_stride(Int<1>{}, Br + KernelConfig<T>::PADDING_SIZE));
#else
  // S_P_sm will be accessed in both mode0-majored (row-majored) and mode1-majored (column-majored) manners,
  // and therefore in the device code, swizzling will be applied to its shared memory area to avoid bank-conflict
  const auto S_P_sm_layout = make_layout(make_shape(get<0>(Q_tile_shape), get<0>(K_tile_shape)));
#endif

  //
  // Preparing for kernel launching
  //

  const auto sm_size_required = (cosize(Q_sm_layout)
                          + cosize(K_sm_layout)
                          + cosize(V_sm_layout)
                          + cosize(l_m_sm_layout) * 2 // 2 for l_sm and m_sm
                          + cosize(S_P_sm_layout)
                          ) * static_cast<int32_t>(sizeof(T));

  dim3 block_dims(KernelConfig<T>::NUM_OF_THREADS);
  dim3 grid_dims(grid_dim_x, grid_dim_y);


#ifdef INTERNAL_TEST
  print("grid = (%u, %u), num of threads = %u, sizeof(T) = %zu, sizeof(L_T) = %zu\n", grid_dims.x, grid_dims.y, block_dims.x, sizeof(T), sizeof(L_T));

  print("b = %d, q = %d, k = %d, d = %d, v_d = %d, Br = %d, Bc = %d, sm_size_required = %d/%d\n",
          b, q, k, d, v_d, Br, Bc, sm_size_required, shared_memory_dest.GetAvailableAmount());
#endif

  const auto kernel_args = std::make_tuple(
    Q, Q_layout, Q_tile_shape, Q_sm_layout,
    K, K_layout, K_tile_shape, K_sm_layout,
    V, V_layout, V_tile_shape, V_sm_layout,
    const_cast<volatile T*>(O), O_layout, O_tile_shape,
    const_cast<volatile L_T*>(l), const_cast<volatile T*>(m), l_m_layout, l_m_tile_shape, l_m_sm_layout,
    S_P_sm_layout,
    dot_scaler,
    const_cast<volatile uint32_t*>(Br_occupancy),
    reference_seq_shape,
    Q_seq_order_map, K_seq_order_map, attention_policy
  );


  if (shared_memory_dest.NeedOptin()) {
    constexpr auto impl_func = &ForwardImpl<T, L_T,
                                            decltype(Q_layout), decltype(Q_tile_shape), decltype(Q_sm_layout),
                                            decltype(K_layout), decltype(K_tile_shape), decltype(K_sm_layout),
                                            decltype(V_layout), decltype(V_tile_shape), decltype(V_sm_layout),
                                            decltype(O_layout), decltype(O_tile_shape),
                                            decltype(l_m_layout), decltype(l_m_tile_shape), decltype(l_m_sm_layout),
                                            decltype(S_P_sm_layout),
                                            remove_reference_t<decltype(reference_seq_shape)>,
                                            remove_reference_t<decltype(Q_seq_order_map)>, remove_reference_t<decltype(K_seq_order_map)>,
                                            remove_reference_t<decltype(attention_policy)>
                                          >;

    assert(cudaError_t::cudaSuccess == cudaFuncSetAttribute(impl_func, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_dest.GetAvailableAmount())
        && "Failed to enable the use of the maximum amount of shared memory available");

    std::apply([&impl_func, &grid_dims, &block_dims, sm_size_required, stream](auto&&... args) {
                impl_func<<<grid_dims, block_dims, sm_size_required, stream>>>(std::forward<decltype(args)>(args)...);
              },
              kernel_args);
  }
  else {
    // It has been found that on one of my systems calling impl_func<<<>>>() directly will result in
    // cudaErrorInvalidDeviceFunction error; since the system in quention only have CC = 6.1
    // which simply does not support the adjustment of the max size of the dynamically allocated shared memory,
    // this case is added as a workaround.
    std::apply([&grid_dims, &block_dims, sm_size_required, stream](auto&&... args) {
                ForwardImpl<<<grid_dims, block_dims, sm_size_required, stream>>>(std::forward<decltype(args)>(args)...);
              },
              kernel_args);
  }

  return cudaGetLastError();
}

template <typename T, typename ReferenceSeqShape, typename SeqOrderMap, typename AttentionPolicy>
cudaError_t FlashAttentionLauncher<T, ReferenceSeqShape, SeqOrderMap, AttentionPolicy>::Backward(
        const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
        const int32_t b,
        const int32_t q, const int32_t k,
        const int32_t d, const int32_t v_d,
        const T* Q, const T* K, const T* V, const T* O, const L_T* l, const T* m,
        const T* dO,
        T* dQ, T* dK, T* dV,
        uint32_t* Br_occupancy,
        const ReferenceSeqShape& reference_seq_shape,
        const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
        const AttentionPolicy& attention_policy) const {

  const auto [Br, Bc] = DetermineConfigurationForBackward<T, L_T>(shared_memory_dest.GetAvailableAmount(), b, q, k, d, v_d);

  T dot_scaler = TypeUtil<T>::RSqrt(static_cast<T>(d));

  #if 0
  T log_alibi_slope_base = TypeUtil<T>::GetNegInfApprox();
  if (alibi_on) {
    const T number_of_heads = static_cast<T>(get<decltype(rank(reference_seq_shape))::value-1>(reference_seq_shape));
    log_alibi_slope_base = TypeUtil<T>::Max(T(-8) / number_of_heads, T(-1));
  }
  #endif

  // The size of the x dimension of the thtread grid
  // is set to the number of Bc sections needed to cover k,
  const auto grid_dim_x = (k + Bc - 1) / Bc;
  // The y dimension of the thread grid is set to b (the batch size)
  const auto grid_dim_y = b;

  //
  // Layouts for global memories
  //

  const auto Q_dQ_layout = make_layout(make_shape(q, d, b));
  const auto K_dK_layout = make_layout(make_shape(k, d, b));
  const auto V_dV_layout = make_layout(make_shape(k, v_d, b));
  const auto O_dO_layout = make_layout(make_shape(q, v_d, b));
  const auto l_m_layout = make_layout(make_shape(q, Int<1>{}, b));

  //
  // Tile shapes
  //

  // The batch dimension (mode 2, indicing starting from 0) is not tiled,
  // and it will be indexed by the y coord of the thread grid
  const auto Q_dQ_tile_shape = make_shape(Br, d, Int<1>{});
  const auto K_dK_tile_shape = make_shape(Bc, d, Int<1>{});
  const auto V_dV_tile_shape = make_shape(Bc, v_d, Int<1>{});
  const auto O_dO_tile_shape = make_shape(Br, v_d, Int<1>{});
  const auto l_m_tile_shape = make_shape(Br, Int<1>{}, Int<1>{});


  //
  // Layouts for shared memories
  //
  const auto Q_sm_layout = make_layout(select<0, 1>(Q_dQ_tile_shape));
  const auto K_dK_sm_layout = make_layout(select<0, 1>(K_dK_tile_shape));
  const auto V_dV_sm_layout = make_layout(select<0, 1>(V_dV_tile_shape));
  const auto l_m_sm_layout = make_layout(select<0, 1>(l_m_tile_shape));

  // P_sm is mode1-majored (column-majored)
  const auto P_sm_layout = make_layout(make_shape(get<0>(Q_dQ_tile_shape), get<0>(K_dK_tile_shape)), LayoutRight{});

#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  const auto padded_Br = Br + KernelConfig<T>::PADDING_SIZE;

  // both O_sm and dO_sm are padded along the Br dimension by PADDING_SIZE
  const auto O_dO_sm_layout = make_layout(select<0, 1>(O_dO_tile_shape), make_stride(Int<1>{}, padded_Br));

  // S_sm is padded along the Br dimension by PADDING_SIZE;
  const auto S_sm_layout = make_layout(make_shape(get<0>(Q_dQ_tile_shape), get<0>(K_dK_tile_shape)), make_stride(Int<1>{}, padded_Br));
#else
  // O_sm and dO_sm will be accessed in both mode0-majored (row-majored) and mode1-majored (column-majored) manners,
  // and therefore in the device code, swizzling will be applied to their shared memory area to avoid bank-conflict;
  // also note that O_sm and S_sm will share the same memory space.
  const auto O_dO_sm_layout = make_layout(select<0, 1>(O_dO_tile_shape));

  // S_sm will be accessed in both mode0-majored (row-majored) and mode1-majored (column-majored) manners,
  // and therefore in the device code, swizzling will be applied to its shared memory area to avoid bank-conflict;
  // also note that O_sm and S_sm will share the same memory space.
  const auto S_sm_layout = make_layout(make_shape(get<0>(Q_dQ_tile_shape), get<0>(K_dK_tile_shape)));
#endif

  //
  // Preparing for kernel launching
  //

  const auto sm_size_required = (cosize(Q_sm_layout)
                              + cosize(K_dK_sm_layout) * 2 // 2 for K_sm and dK_sm
                              + cosize(V_dV_sm_layout) * 2 // 2 for V_sm and dV_sm
                              + cosize(O_dO_sm_layout)
                              + cosize(P_sm_layout)
                              + max(cosize(S_sm_layout), cosize(O_dO_sm_layout)) // for the memory area shared by S_sm and O_sm
                              + cosize(l_m_sm_layout)
                              ) * static_cast<int32_t>(sizeof(T))
                          + cosize(l_m_sm_layout) * static_cast<int32_t>(sizeof(L_T)); // for l_sm, of which the data type is L_T instead of T

  dim3 block_dims(KernelConfig<T>::NUM_OF_THREADS);
  dim3 grid_dims(grid_dim_x, grid_dim_y);

#ifdef INTERNAL_TEST
  print("grid = (%u, %u), num of threads = %u, sizeof(T) = %zu, sizeof(L_T) = %zu\n", grid_dims.x, grid_dims.y, block_dims.x, sizeof(T), sizeof(L_T));

  print("b = %d, q = %d, k = %d, d = %d, v_d = %d, Br = %d, Bc = %d, sm_size_required = %d/%d\n",
          b, q, k, d, v_d, Br, Bc, sm_size_required, shared_memory_dest.GetAvailableAmount());
#endif

  const auto kernel_args = std::make_tuple(
    Q, Q_dQ_layout, Q_dQ_tile_shape, Q_sm_layout,
    K, K_dK_layout, K_dK_tile_shape, K_dK_sm_layout,
    V, V_dV_layout, V_dV_tile_shape, V_dV_sm_layout,
    O, dO, O_dO_layout, O_dO_tile_shape, O_dO_sm_layout,
    l, m, l_m_layout, l_m_tile_shape, l_m_sm_layout,
    const_cast<volatile T*>(dQ), dK, dV,
    P_sm_layout, S_sm_layout,
    dot_scaler,
    const_cast<volatile uint32_t*>(Br_occupancy),
    reference_seq_shape,
    Q_seq_order_map, K_seq_order_map, attention_policy
  );

  if (shared_memory_dest.NeedOptin()) {
    constexpr auto impl_func = &BackwardImpl<T, L_T,
                                              decltype(Q_dQ_layout), decltype(Q_dQ_tile_shape), decltype(Q_sm_layout),
                                              decltype(K_dK_layout), decltype(K_dK_tile_shape), decltype(K_dK_sm_layout),
                                              decltype(V_dV_layout), decltype(V_dV_tile_shape), decltype(V_dV_sm_layout),
                                              decltype(O_dO_layout), decltype(O_dO_tile_shape), decltype(O_dO_sm_layout),
                                              decltype(l_m_layout), decltype(l_m_tile_shape), decltype(l_m_sm_layout),
                                              decltype(P_sm_layout), decltype(S_sm_layout),
                                              remove_reference_t<decltype(reference_seq_shape)>,
                                              remove_reference_t<decltype(Q_seq_order_map)>, remove_reference_t<decltype(K_seq_order_map)>,
                                              remove_reference_t<decltype(attention_policy)>
                                            >;
    assert(cudaError_t::cudaSuccess == cudaFuncSetAttribute(impl_func, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_dest.GetAvailableAmount())
        && "Failed to enable the use of the maximum amount of shared memory available");

    std::apply([&impl_func, &grid_dims, &block_dims, sm_size_required, stream](auto&&... args) {
                impl_func<<<grid_dims, block_dims, sm_size_required, stream>>>(std::forward<decltype(args)>(args)...);
              },
              kernel_args);
  }
  else {
    // It has been found that on one of my systems calling impl_func<<<>>>() directly will result in
    // cudaErrorInvalidDeviceFunction error; since the system in quention only have CC = 6.1
    // which simply does not support the adjustment of the max size of the shared memory allocated dynamically,
    // this case is added as a workaround.
    std::apply([&grid_dims, &block_dims, sm_size_required, stream](auto&&... args) {
                BackwardImpl<<<grid_dims, block_dims, sm_size_required, stream>>>(std::forward<decltype(args)>(args)...);
              },
              kernel_args);
  }

  return cudaGetLastError();
}


using Seq1dOrderMap = cute::Tensor<
                                    cute::ViewEngine<cute::ArithmeticTupleIterator<cute::ArithmeticTuple<int32_t>>>,
                                    cute::Layout<cute::tuple<int32_t>, cute::tuple<cute::ScaledBasis<int32_t, 0>>>
                                  >;
using Seq2dOrderMap = cute::Tensor<
                                    cute::ViewEngine<cute::ArithmeticTupleIterator<cute::ArithmeticTuple<int32_t>>>,
                                    cute::Layout<cute::tuple<int32_t, int32_t>, cute::tuple<cute::ScaledBasis<int32_t, 0>, cute::ScaledBasis<int32_t, 0>>>
                                  >;

using ReferenceSeqShape1d = cute::tuple<int32_t>;
using ReferenceSeqShape2d = cute::tuple<int32_t, int32_t>;

#define INSTANTIATE_LAUNCHER(T, ReferenceSeqShape, SeqOrderMap, AttentionPolicy) \
  template struct FlashAttentionLauncher<T, ReferenceSeqShape, SeqOrderMap, AttentionPolicy>;

#if defined(GOOGLE_CUDA)

#define HALF_TYPE_ITERATOR half
#include "macro_util.h"

#define ITERATE_SEQ_DEFINITIONS(func, ...) \
  func(__VA_ARGS__, ReferenceSeqShape1d, Seq1dOrderMap) \
  func(__VA_ARGS__, ReferenceSeqShape2d, Seq2dOrderMap)

#define ITERATE_ATTENTION_POLICY(func, ...) \
  func(__VA_ARGS__, FullAttentionPolicy) \
  func(__VA_ARGS__, CausalAttentionPolicy) \
  func(__VA_ARGS__, LocalAttentionPolicy)

// This will iterate through available combinations of data types, sequence definitions, and attention policies,
// so as to explicitly instantiate functors supporting those configurations
ITERATE_TYPES(ITERATE_SEQ_DEFINITIONS, ITERATE_ATTENTION_POLICY, INSTANTIATE_LAUNCHER)

#elif defined(INTERNAL_TEST)

INSTANTIATE_LAUNCHER(INTERNAL_TEST, ReferenceSeqShape1d, Seq1dOrderMap, FullAttentionPolicy)

#endif

} // namespace cuda_launch
