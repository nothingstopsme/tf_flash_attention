#ifndef __SYNC_METHODS_H__
#define __SYNC_METHODS_H__

#include <optional>
#include <cute/tensor.hpp>

namespace tensorflow {
class TensorShape;
}

struct SequenceDescriptor {
  // Note that while int64_t is used to store the number of dimensions,
  // currently the sizes of each dimension, and therefore indices of them, are limited to the range representable by int32_t
  using Vector = std::vector<int32_t>;
  SequenceDescriptor(const int64_t dims) {
    shape.reserve(dims);
    stride.reserve(dims);
    offset.reserve(dims);
  }

  SequenceDescriptor(SequenceDescriptor&& other)
  : shape(std::move(other.shape)),
    stride(std::move(other.stride)),
    offset(std::move(other.offset)) {
  }

  Vector shape;
  Vector stride;
  Vector offset;
};

struct SequenceDescriptorPack {
  SequenceDescriptorPack(const int64_t dims)
  : Q_desc(dims), K_desc(dims) {
    reference_shape.reserve(dims);
  }

  SequenceDescriptor::Vector reference_shape;
  SequenceDescriptor Q_desc;
  SequenceDescriptor K_desc;
};

using MethodImplPtr = SequenceDescriptorPack (*)(const tensorflow::TensorShape&, const tensorflow::TensorShape&);

template <int32_t SequenceDims>
class SyncMethod {
 public:
  SyncMethod(const MethodImplPtr method_impl_ptr)
  : _method_impl_ptr(method_impl_ptr) {
  }

  inline auto operator()(const tensorflow::TensorShape& Q_seq_shape, const tensorflow::TensorShape& K_seq_shape) const {
    return GenerateOrderMap(_method_impl_ptr(Q_seq_shape, K_seq_shape));
  }

  static inline auto GenerateOrderMap(const SequenceDescriptorPack& pack) {
    using namespace cute;

    assert(SequenceDims == pack.reference_shape.size()
          && "The dimensionality of the given reference_shape does not match SequenceDims");

    const auto reference_seq_shape = wrap(make_int_tuple<SequenceDims>(pack.reference_shape.data(), pack.reference_shape.size(), Int<0>{}));

    return make_tuple(reference_seq_shape, GenerateOrderMap(reference_seq_shape, pack.Q_desc), GenerateOrderMap(reference_seq_shape, pack.K_desc));
  }


 private:
  template <typename ReferenceShape>
  static inline auto GenerateOrderMap(const ReferenceShape& reference_seq_shape, const SequenceDescriptor& seq_desc) {
    using namespace cute;

    assert(SequenceDims == seq_desc.shape.size()
          && SequenceDims == seq_desc.stride.size()
          && SequenceDims == seq_desc.offset.size()
          && "The dimensionality of the given sequence definition does not match SequenceDims");

    const auto seq_shape = wrap(make_int_tuple<SequenceDims>(seq_desc.shape.data(), seq_desc.shape.size(), Int<0>{}));
    const auto seq_stride = wrap(make_int_tuple<SequenceDims>(seq_desc.stride.data(), seq_desc.stride.size(), Int<0>{}));
    const auto seq_offset = wrap(make_int_tuple<SequenceDims>(seq_desc.offset.data(), seq_desc.offset.size(), Int<0>{}));

    Tensor order_map = make_identity_tensor(make_shape(size(reference_seq_shape))).compose(make_layout(reference_seq_shape));

    return flatten(zipped_divide(flatten(zipped_divide(order_map, seq_stride)(seq_offset, _)), seq_shape)(_, Int<0>{}));
  }

  const MethodImplPtr _method_impl_ptr;

};

class SyncMethods {
 public:
  SyncMethods() = delete;


  template <int32_t SequenceDims>
  static void Lookup(const std::string& name, std::optional<SyncMethod<SequenceDims>>& functor) {
    const auto iter = _METHOD_TABLE.find(name);
    if (iter != _METHOD_TABLE.end())
      functor.emplace(iter->second);
  }


 private:
  static const std::unordered_map<std::string, MethodImplPtr> _METHOD_TABLE;

};

#endif // __SYNC_METHODS_H__
