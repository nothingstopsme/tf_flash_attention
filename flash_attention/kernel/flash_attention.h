
#ifndef __FLASH_ATTENTION_H__
#define __FLASH_ATTENTION_H__

#include <cuda_fp16.h>
#include "cute/layout.hpp"

class AttentionPolicy {
 protected:
  template <typename ReferenceSeqShape, int Dim = 0>
  CUTE_HOST_DEVICE
  static constexpr auto MapToCoords(const int index, const ReferenceSeqShape& reference_seq_shape,
                                    const int log2_stride = 0) {

    if constexpr (Dim < decltype(cute::rank(reference_seq_shape))::value) {
      const auto s = cute::get<Dim>(reference_seq_shape);
      #ifdef __CUDA_ARCH__
      const auto log2_s = __ffs(s) - 1;
      #else
      const auto log2_s = static_cast<int>(std::log2(static_cast<double>(s)));
      #endif
      const auto mask = s - 1;

      return cute::tuple_cat(cute::wrap((index >> log2_stride) & mask), MapToCoords<ReferenceSeqShape, Dim+1>(index, reference_seq_shape, log2_stride + log2_s));
    } else {
      return cute::tuple<>{};
    }
  }

  template <typename Coords, typename ReferenceSeqShape, int Dim = 0>
  CUTE_HOST_DEVICE
  static constexpr int MapToOrder(const Coords& coords, const ReferenceSeqShape& reference_seq_shape,
                                    const int log2_stride = 0) {

    if constexpr (Dim < decltype(cute::rank(reference_seq_shape))::value) {
      const auto s = cute::get<Dim>(reference_seq_shape);
      const auto c = cute::get<Dim>(coords);
      #ifdef __CUDA_ARCH__
      const auto log2_s = __ffs(s) - 1;
      #else
      const auto log2_s = static_cast<int>(std::log2(static_cast<double>(s)));
      #endif

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
                            const int min_Q_order, const int max_Q_order,
                            const int min_K_order, const int max_K_order) const {
    return false;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int Q_order, const int K_order) const {
    return true;
  }
};


class CausalAttentionPolicy : public AttentionPolicy {
 public:
  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool IsSkipped(const ReferenceSeqShape& reference_seq_shape,
                            const int min_Q_order, const int max_Q_order,
                            const int min_K_order, const int max_K_order) const {
    return max_Q_order < min_K_order;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int Q_order, const int K_order) const {
    return Q_order >= K_order;
  }
};

class LocalAttentionPolicy : public AttentionPolicy {
 public:
  CUTE_HOST_DEVICE
  LocalAttentionPolicy(const int window_size, const int log2_stride_size, const bool is_causal)
  : _window_size(window_size), _log2_stride_size(log2_stride_size),
    _strided_window_size(_window_size << _log2_stride_size),
    _remainder_mask((uint32_t(1) << _log2_stride_size) - 1) {

    assert(_log2_stride_size < 31 && "stride size is too big; please make sure the stride size/window size is within the range representable by int");
    if (is_causal) {
      _look_ahead_size = 1; // 1 for looking at self
    } else {
      _look_ahead_size = _strided_window_size;
    }
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool IsSkipped(const ReferenceSeqShape& reference_seq_shape,
                              const int min_Q_order, const int max_Q_order,
                              const int min_K_order, const int max_K_order) const {
    // Assumption:
    // order indices are always valid with respect to the given reference_seq_shape,
    // i.e. indices < size(reference_seq_shape)
    const auto min_Q_coords = MapToCoords(min_Q_order, reference_seq_shape);
    const auto max_Q_coords = MapToCoords(max_Q_order, reference_seq_shape);

    const auto min_window_entry_coords = cute::transform(min_Q_coords, [this](const auto& c) { return max(c - _strided_window_size + 1, long(0)); });
    const auto max_window_entry_coords = cute::transform(max_Q_coords, reference_seq_shape, [this](const auto& c, const auto& limit) { return min(c + _look_ahead_size, limit) - 1; });

    const auto min_window_entry_order = MapToOrder(min_window_entry_coords, reference_seq_shape);
    const auto max_window_entry_order = MapToOrder(max_window_entry_coords, reference_seq_shape);
    return max_K_order < min_window_entry_order || min_K_order > max_window_entry_order;
  }

  template <typename ReferenceSeqShape>
  CUTE_HOST_DEVICE
  constexpr bool Check(const ReferenceSeqShape& reference_seq_shape,
                        const int Q_order, const int K_order) const {
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
  int _window_size;
  int _log2_stride_size;
  int _strided_window_size;
  int _remainder_mask;
  int _look_ahead_size;

};


namespace cuda_launch {


class SharedMemoryDescriptor {
 public:
  explicit SharedMemoryDescriptor(const int basic_amount, const int optin_amount) {
    if (optin_amount > basic_amount) {
      available_amount_ = optin_amount;
      need_optin_ = true;
    }
    else {
      available_amount_ = basic_amount;
      need_optin_ = false;
    }
  }

  inline int GetAvailableAmount() const {
    return available_amount_;
  }

  inline bool NeedOptin() const {
    return need_optin_;
  }

 private:
  int available_amount_;
  bool need_optin_;
};

template <typename T>
struct L_TypeMapping { using type = T; };

template <>
struct L_TypeMapping<half> { using type = float; };

template <typename T>
struct KernelConfig {

  using L_T = typename L_TypeMapping<T>::type;
  static constexpr int NUM_OF_T_PER_TRANSACTION = 128 / sizeof(T);
  static constexpr int NUM_OF_THREADS = 256;
  static constexpr int BR_SIZE = std::max(32, NUM_OF_T_PER_TRANSACTION);
  static constexpr int PADDING_SIZE = sizeof(T) == 2 ? 2 : 1;
  static constexpr int MODE1_MAJOR_SIZE = 32;


  using Mode0MajorThreadLayout = decltype(cute::make_layout(cute::make_shape(cute::Int<BR_SIZE>{}, cute::Int<NUM_OF_THREADS / BR_SIZE>{})));

  using Mode1MajorThreadLayout = decltype(cute::make_layout(cute::make_shape(cute::Int<NUM_OF_THREADS / MODE1_MAJOR_SIZE>{}, cute::Int<MODE1_MAJOR_SIZE>{}), cute::LayoutRight{}));

};


template <typename T, typename ReferenceSeqShape, typename SeqOrderMap, typename AttentionPolicy>
class FlashAttentionLauncher {
 public:
  using L_T = typename KernelConfig<T>::L_T;

  cudaError_t Forward(const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
                const int64_t b,
                const int64_t q, const int64_t k,
                const int64_t d, const int64_t v_d,
                const T* Q, const T* K, const T* V,
                T* O, L_T* l, T* m,
                uint32_t* Br_occupancy,
                const ReferenceSeqShape& reference_seq_shape,
                const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
                const AttentionPolicy& attention_policy) const;

  cudaError_t Backward(const cudaStream_t stream, const SharedMemoryDescriptor& shared_memory_dest,
                const int64_t b,
                const int64_t q, const int64_t k,
                const int64_t d, const int64_t v_d,
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
                              const int64_t b,
                              const int64_t q, const int64_t k,
                              const int64_t d, const int64_t v_d,
                              const ReferenceSeqShape& reference_seq_shape,
                              const SeqOrderMap& Q_seq_order_map, const SeqOrderMap& K_seq_order_map,
                              const AttentionPolicy& attention_policy,
                              float& flops) const;
};

} // namespace cuda_launch

#endif // __FLASH_ATTENTION_H__
