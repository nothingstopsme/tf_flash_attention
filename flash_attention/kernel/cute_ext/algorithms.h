#ifndef __CUTE_EXT_ALGORITHMS_H__
#define __CUTE_EXT_ALGORITHMS_H__

#include <cute/tensor.hpp>

namespace cute_ext {

template <typename Integer>
CUTE_HOST_DEVICE
constexpr Integer log2i(Integer n) {
  #ifdef __CUDA_ARCH__
  return __ffs(n) - 1;
  #else
  return ((n < 2) ? 0 : 1 + log2i(n >> 1));
  #endif
}

template <typename Pred,
          typename Tensor>
__device__
inline constexpr void FillIf(
                  const Pred& pred,
                  Tensor&& tensor,
                  const typename std::remove_reference_t<std::remove_cv_t<Tensor>>::value_type& filling) {
  CUTE_UNROLL
  for (int32_t i = 0; i < cute::size(tensor); ++i) {
    if (pred(i))
      tensor(i) = filling;
  }
}

template <typename CopyOp,
          typename Pred,
          typename SrcTensor,
          typename DstTensor>
__device__
inline constexpr void CopyIf(
                  const CopyOp& copy_op,
                  const Pred& pred,
                  const SrcTensor& src,
                  DstTensor&& dst) {
  CUTE_UNROLL
  for (int32_t i = 0; i < cute::size(src); ++i) {
    if (pred(i))
      copy_op.copy(src(i), dst(i));
  }
}

template <typename CopyOp,
          typename SrcPred,
          typename SrcTensor,
          typename DstPred,
          typename DstTensor>
__device__
inline constexpr void CopyOrFill(
                const CopyOp& copy_op,
                const SrcPred& src_pred,
                const SrcTensor& src,
                const DstPred& dst_pred,
                DstTensor&& dst,
                const typename std::remove_reference_t<std::remove_cv_t<DstTensor>>::value_type& filling) {
  CUTE_UNROLL
  for (int32_t i = 0; i < cute::size(src); ++i) {
    if (dst_pred(i)) {
      if (src_pred(i))
        copy_op.copy(src(i), dst(i));
      else
        dst(i) = filling;
    }
  }
}

} // namespace cute_ext


#endif // __CUTE_EXT_ALGORITHMS_H__
