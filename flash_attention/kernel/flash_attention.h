
#ifndef __FLASH_ATTENTION_H__
#define __FLASH_ATTENTION_H__

#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include "cute_ext/algorithms.h"

class AttentionPolicy {
 protected:
  template <typename ReferenceSeqShape, int Dim = 0>
  CUTE_HOST_DEVICE
  static constexpr auto MapToCoords(const int32_t index, const ReferenceSeqShape& reference_seq_shape,
                                    const int32_t log2_stride = 0) {

    if constexpr (Dim < decltype(cute::rank(reference_seq_shape))::value) {
      const auto s = cute::get<Dim>(reference_seq_shape);
      const auto log2_s = cute_ext::log2i(s);
      const auto mask = s - 1;

      return cute::tuple_cat(cute::wrap((index >> log2_stride) & mask), MapToCoords<ReferenceSeqShape, Dim+1>(index, reference_seq_shape, log2_stride + log2_s));
    } else {
      return cute::tuple<>{};
    }
  }

  template <typename Coords, typename ReferenceSeqShape, int Dim = 0>
  CUTE_HOST_DEVICE
  static constexpr int MapToOrder(const Coords& coords, const ReferenceSeqShape& reference_seq_shape,
                                    const int32_t log2_stride = 0) {

    if constexpr (Dim < decltype(cute::rank(reference_seq_shape))::value) {
      const auto s = cute::get<Dim>(reference_seq_shape);
      const auto c = cute::get<Dim>(coords);
      const auto log2_s = cute_ext::log2i(s);

      return (c << log2_stride) + MapToOrder<Coords, ReferenceSeqShape, Dim+1>(coords, reference_seq_shape, log2_stride + log2_s);
    } else {
      return 0;
    }
  }

};

class FullAttentionPolicy : public AttentionPolicy {
 public:
  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool IsSkipped(const ReferenceSeqShape& reference_seq_shape,
                            const int32_t min_Q_order, const int32_t max_Q_order,
                            const int32_t min_K_order, const int32_t max_K_order) const {
    return false;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int32_t Q_order, const int32_t K_order) const {
    return true;
  }
};


class CausalAttentionPolicy : public AttentionPolicy {
 public:
  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool IsSkipped(const ReferenceSeqShape& reference_seq_shape,
                            const int32_t min_Q_order, const int32_t max_Q_order,
                            const int32_t min_K_order, const int32_t max_K_order) const {
    return max_Q_order < min_K_order;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int32_t Q_order, const int32_t K_order) const {
    return Q_order >= K_order;
  }
};

class LocalAttentionPolicy : public AttentionPolicy {
 public:
  CUTE_HOST_DEVICE
  LocalAttentionPolicy(const int32_t window_size, const int32_t log2_stride_size, const bool is_causal)
  : _window_size(window_size), _log2_stride_size(log2_stride_size),
    _strided_window_size(_window_size << _log2_stride_size),
    _remainder_mask((int32_t(1) << _log2_stride_size) - 1) {

    assert(_log2_stride_size < 31 && _strided_window_size >= _window_size && "stride size is too big; please make sure the stride size/window size is within the range representable by int32_t");
    if (is_causal) {
      _look_ahead_size = 1; // 1 for looking at self
    } else {
      _look_ahead_size = _strided_window_size;
    }
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool IsSkipped(const ReferenceSeqShape& reference_seq_shape,
                              const int32_t min_Q_order, const int32_t max_Q_order,
                              const int32_t min_K_order, const int32_t max_K_order) const {
    // Assumption:
    // order indices are always valid with respect to the given reference_seq_shape,
    // i.e. indices < size(reference_seq_shape)
    const auto min_Q_coords = MapToCoords(min_Q_order, reference_seq_shape);
    const auto max_Q_coords = MapToCoords(max_Q_order, reference_seq_shape);

    const auto min_window_entry_coords = cute::transform(min_Q_coords, [this](const auto& c) { return max(c - _strided_window_size + 1, int32_t(0)); });
    const auto max_window_entry_coords = cute::transform(max_Q_coords, reference_seq_shape, [this](const auto& c, const auto& limit) { return min(c + _look_ahead_size, limit) - 1; });

    const auto min_window_entry_order = MapToOrder(min_window_entry_coords, reference_seq_shape);
    const auto max_window_entry_order = MapToOrder(max_window_entry_coords, reference_seq_shape);
    return max_K_order < min_window_entry_order || min_K_order > max_window_entry_order;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int32_t Q_order, const int32_t K_order) const {
    constexpr cute::logical_and and_func{};

    if (_look_ahead_size == 1 && Q_order < K_order)
      return false;

    const auto Q_coords = MapToCoords(Q_order, reference_seq_shape);
    const auto K_coords = MapToCoords(K_order, reference_seq_shape);



    return cute::fold_first(cute::transform(Q_coords, K_coords, [this](const auto& Q_coord, const auto& K_coord) {
                                auto diff = abs(Q_coord - K_coord);
                                if ((diff & _remainder_mask) == 0) {
                                  return (diff >> _log2_stride_size) < _window_size;
                                }
                                return false;
                              }),
                            and_func);

  }

 private:
  int32_t _window_size;
  int32_t _log2_stride_size;
  int32_t _strided_window_size;
  int32_t _remainder_mask;
  int32_t _look_ahead_size;

};


namespace cuda_launch {


class SharedMemoryDescriptor {
 public:
  explicit SharedMemoryDescriptor(const int32_t basic_amount, const int32_t optin_amount) {
    if (optin_amount > basic_amount) {
      available_amount_ = optin_amount;
      need_optin_ = true;
    }
    else {
      available_amount_ = basic_amount;
      need_optin_ = false;
    }
  }

  inline int32_t GetAvailableAmount() const {
    return available_amount_;
  }

  inline bool NeedOptin() const {
    return need_optin_;
  }

 private:
  int32_t available_amount_;
  bool need_optin_;
};

template <typename T>
struct L_TypeMapping { using type = T; };

template <>
struct L_TypeMapping<half> { using type = float; };

template <typename T>
struct KernelConfig {
 private:
  static constexpr int _BANK_SIZE = 4;
  static constexpr int _SIZE_OF_TRANSACTION = 32 * _BANK_SIZE;
  static constexpr int _MODE1_MAJOR_SIZE = 32;


  static constexpr int _CONTINUOUS_BYTES = std::max(static_cast<int>(sizeof(T)), _BANK_SIZE);

 public:
  using L_T = typename L_TypeMapping<T>::type;
  static constexpr int NUM_OF_THREADS = 256;
  static constexpr int BR_SIZE = std::max(32, static_cast<int>(_SIZE_OF_TRANSACTION / sizeof(T)));


#ifndef BANK_CONFLICT_FREE_BY_SWIZZLING
  static constexpr int PADDING_SIZE = sizeof(T) == 2 ? 2 : 1;
#else
  // BBits of the swizzle configuration below is set to the num of groups of continuous bytes making up one transaction,
  // which is the number of group addresses that can be swizzled about;
  // while SShift is set to the number of groups of continuous bytes along Br dimension,
  // which means a new swizzle mapping (i.e. with a differnt Y mask) starts every that number of byte groups
  using SmSwizzler = cute::Swizzle<cute_ext::log2i(_SIZE_OF_TRANSACTION / _CONTINUOUS_BYTES), cute_ext::log2i(_CONTINUOUS_BYTES), cute_ext::log2i(BR_SIZE * sizeof(T) / _CONTINUOUS_BYTES)>;
#endif


  using Mode0MajorThreadLayout = decltype(cute::make_layout(cute::make_shape(cute::Int<BR_SIZE>{}, cute::Int<NUM_OF_THREADS / BR_SIZE>{})));

  using Mode1MajorThreadLayout = decltype(cute::make_layout(cute::make_shape(cute::Int<NUM_OF_THREADS / _MODE1_MAJOR_SIZE>{}, cute::Int<_MODE1_MAJOR_SIZE>{}), cute::LayoutRight{}));
};


template <typename T, typename ReferenceSeqShape, typename SeqOrderMap, typename AttentionPolicy>
class FlashAttentionLauncher {
 public:
  using L_T = typename KernelConfig<T>::L_T;

  cudaError_t Forward(const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
                const int32_t b,
                const int32_t q, const int32_t k,
                const int32_t d, const int32_t v_d,
                const T* Q, const T* K, const T* V,
                T* O, L_T* l, T* m,
                uint32_t* Br_occupancy,
                const ReferenceSeqShape& reference_seq_shape,
                const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
                const AttentionPolicy& attention_policy) const;

  cudaError_t Backward(const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
                const int32_t b,
                const int32_t q, const int32_t k,
                const int32_t d, const int32_t v_d,
                const T* Q, const T* K, const T* V, const T* O, const L_T* l, const T* m,
                const T* dO,
                T* dQ, T* dK, T* dV,
                uint32_t* Br_occupancy,
                const ReferenceSeqShape& reference_seq_shape,
                const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
                const AttentionPolicy& attention_policy) const;

  static inline constexpr auto ComputeNumOfBrSections(const int length) {
    return (length + KernelConfig<T>::BR_SIZE - 1) / KernelConfig<T>::BR_SIZE;
  }

  void EstimateForwardFlops(const SharedMemoryDescriptor& shared_memory_dest,
                              const int32_t b,
                              const int32_t q, const int32_t k,
                              const int32_t d, const int32_t v_d,
                              const ReferenceSeqShape& reference_seq_shape,
                              const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
                              const AttentionPolicy& attention_policy,
                              float& flops) const;
};

} // namespace cuda_launch

#endif // __FLASH_ATTENTION_H__
