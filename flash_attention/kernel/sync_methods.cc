
#include "sync_methods.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

static SequenceDescriptorPack SyncNoneFront(const TensorShape& Q_seq_shape, const TensorShape& K_seq_shape) {
  // Q_seq_shape and K_seq_shape must have the same rank
  SequenceDescriptorPack pack(Q_seq_shape.dims());

  // Inserting dimension info in reversing order
  for (int64_t dim = Q_seq_shape.dims()-1; dim >= 0; --dim) {
    const int64_t Q_dim = Q_seq_shape.dim_size(dim);
    const int64_t K_dim = K_seq_shape.dim_size(dim);
    const int64_t max_dim = Q_dim > K_dim ? Q_dim : K_dim;

    // For the purpose of efficient integer arithmetic,
    // ref_dim is set to a power of 2 >= max_dim
    const int64_t ref_dim = static_cast<int64_t>(std::pow(2.0, std::ceil(std::log2(static_cast<double>(max_dim)))));

    pack.reference_shape.push_back(ref_dim);

    pack.Q_desc.shape.push_back(Q_dim);
    pack.Q_desc.stride.push_back(1);
    pack.Q_desc.offset.push_back(0);

    pack.K_desc.shape.push_back(K_dim);
    pack.K_desc.stride.push_back(1);
    pack.K_desc.offset.push_back(0);

  }

  return pack;

}



static SequenceDescriptorPack SyncScaleFront(const TensorShape& Q_seq_shape, const TensorShape& K_seq_shape) {
  // Q_seq_shape and K_seq_shape must have the same rank
  SequenceDescriptorPack pack(Q_seq_shape.dims());

  // Inserting dimension info in reversing order
  for (int64_t dim = Q_seq_shape.dims()-1; dim >= 0; --dim) {
    const int64_t Q_dim = Q_seq_shape.dim_size(dim);
    const int64_t K_dim = K_seq_shape.dim_size(dim);
    const int64_t max_dim = Q_dim > K_dim ? Q_dim : K_dim;

    // For the purpose of efficient integer arithmetic,
    // ref_dim is set to a power of 2 >= max_dim
    const int64_t ref_dim = static_cast<int64_t>(std::pow(2.0, std::ceil(std::log2(static_cast<double>(max_dim)))));

    pack.reference_shape.push_back(ref_dim);

    pack.Q_desc.shape.push_back(Q_dim);
    pack.Q_desc.stride.push_back(max_dim / Q_dim);
    pack.Q_desc.offset.push_back(0);

    pack.K_desc.shape.push_back(K_dim);
    pack.K_desc.stride.push_back(max_dim / K_dim);
    pack.K_desc.offset.push_back(0);

  }

  return pack;

}


static SequenceDescriptorPack SyncScaleEnd(const TensorShape& Q_seq_shape, const TensorShape& K_seq_shape) {
  // Q_seq_shape and K_seq_shape must have the same rank
  SequenceDescriptorPack pack(Q_seq_shape.dims());

  // Inserting dimension info in reversing order
  for (int64_t dim = Q_seq_shape.dims()-1; dim >= 0; --dim) {
    const int64_t Q_dim = Q_seq_shape.dim_size(dim);
    const int64_t K_dim = K_seq_shape.dim_size(dim);
    const int64_t max_dim = Q_dim > K_dim ? Q_dim : K_dim;

    // For the purpose of efficient integer arithmetic,
    // ref_dim is set to a power of 2 >= max_dim
    const int64_t ref_dim = static_cast<int64_t>(std::pow(2.0, std::ceil(std::log2(static_cast<double>(max_dim)))));

    pack.reference_shape.push_back(ref_dim);

    pack.Q_desc.shape.push_back(Q_dim);
    pack.Q_desc.stride.push_back(max_dim / Q_dim);
    pack.Q_desc.offset.push_back(pack.Q_desc.stride.back()-1);

    pack.K_desc.shape.push_back(K_dim);
    pack.K_desc.stride.push_back(max_dim / K_dim);
    pack.K_desc.offset.push_back(pack.K_desc.stride.back()-1);

  }

  return pack;
}

const std::unordered_map<std::string, MethodImplPtr> SyncMethods::_METHOD_TABLE = {
  {"none_front", &SyncNoneFront},
  {"scale_front", &SyncScaleFront},
  {"scale_end", &SyncScaleEnd},
};


