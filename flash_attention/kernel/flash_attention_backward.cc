

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "type_mapping.h"
#include "flash_attention.h"
#include "sync_methods.h"
#include "tf_type_declaration.h"

#define HALF_TYPE_ITERATOR Eigen::half
#include "macro_util.h"

using namespace cuda_launch;

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

namespace {

static inline absl::Status InferShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle Q_shape = c->input(0);
  shape_inference::ShapeHandle K_shape = c->input(1);
  shape_inference::ShapeHandle V_shape = c->input(2);

  c->set_output(0, Q_shape);
  c->set_output(1, K_shape);
  c->set_output(2, V_shape);

  return OkStatus();

}

} // anonymous namespace

#define REGISTER_TF_OPS(unused, sequence_dims) \
  REGISTER_OP("FullAttentionBackward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: float") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape); \
  \
  REGISTER_OP("CausalAttentionBackward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: float") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape); \
  \
  REGISTER_OP("LocalAttentionBackward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: float") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Attr("window_size: int >= 1") \
    .Attr("log2_stride_size: int >= 0") \
    .Attr("is_causal: bool") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape); \
  \
  REGISTER_OP("FullAttentionBackward" #sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: T") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape); \
  \
  REGISTER_OP("CausalAttentionBackward" # sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: T") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape); \
  \
  REGISTER_OP("LocalAttentionBackward" # sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Input("o: T") \
    .Input("l: T") \
    .Input("m: T") \
    .Input("d_o: T") \
    .Attr("sync_mode: string") \
    .Attr("window_size: int >= 1") \
    .Attr("log2_stride_size: int >= 0") \
    .Attr("is_causal: bool") \
    .Output("d_q: T") \
    .Output("d_k: T") \
    .Output("d_v: T") \
    .SetShapeFn(&InferShape);

// What ITERATE_SEQUENCE_DIMS() does internally is to take
// its first input as the macro function to be expanded,
// and passing a list of arguments containing the rest of inputs appended
// with a dimension value. Therefore in this case REGISTER_TF_OPS() will receive
// two arguments: the first one is empty (expanded to nothing),
// while the second one is that dimension value
ITERATE_SEQUENCE_DIMS(REGISTER_TF_OPS,)

template <typename T, int SequenceDims, typename AttentionPolicy>
class FlashAttentionBackwardBase : public OpKernel {
 private:
  std::optional<SyncMethod<SequenceDims>> _sync_method;

 protected:
  std::optional<AttentionPolicy> _attention_policy;

 public:
  explicit FlashAttentionBackwardBase(OpKernelConstruction* cons, const char* fixed_sync_mode = nullptr)
  : OpKernel(cons) {
    static_assert(SequenceDims >= 1, "SequenceDims should be >= 1");

    std::string sync_mode;
    if (fixed_sync_mode)
      sync_mode.assign(fixed_sync_mode);
    else
      OP_REQUIRES_OK(cons, cons->GetAttr("sync_mode", &sync_mode));

    SyncMethods::Lookup(sync_mode, _sync_method);
    OP_REQUIRES(cons, _sync_method.has_value(),
                errors::InvalidArgument("Unsupported sync_mode: ", sync_mode));

  }

	void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context,
                _attention_policy.has_value(),
                errors::Internal("attention_policy has set to be set"));



    const Tensor& Q = context->input(0);
    const Tensor& K = context->input(1);
    const Tensor& V = context->input(2);
    const Tensor& O = context->input(3);
    const Tensor& l = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& dO = context->input(6);


    OP_REQUIRES(
      context, Q.dims() == K.dims() && K.dims() == V.dims() && V.dims() == O.dims() && O.dims() == dO.dims(),
      errors::InvalidArgument("The number of dimensions of Q, K, V, O, and dO should be equal"));

    OP_REQUIRES(
      context, l.dims() == m.dims() && m.dims() == Q.dims()-1,
      errors::InvalidArgument("The number of dimensions of l and m should be equal to the one of Q minus 1"));


    OP_REQUIRES(
      context, Q.dims() >= SequenceDims+2,
      errors::InvalidArgument("The number of dimensions of Q, K, V, O, and dO should be >= ", SequenceDims+2));

    int channel_axis = Q.dims() - SequenceDims - 1;


    int64_t Q_ch = Q.dim_size(channel_axis);
    int64_t K_ch = K.dim_size(channel_axis);
    int64_t V_ch = V.dim_size(channel_axis);
    int64_t O_ch = O.dim_size(channel_axis);


    TensorShape Q_batch_shape = Q.shape(),  Q_seq_shape = Q.shape();
    Q_batch_shape.RemoveDimRange(channel_axis, Q.dims());
    Q_seq_shape.RemoveDimRange(0, channel_axis+1);
    TensorShape K_batch_shape = K.shape(), K_seq_shape = K.shape();
    K_batch_shape.RemoveDimRange(channel_axis, K.dims());
    K_seq_shape.RemoveDimRange(0, channel_axis+1);
    TensorShape V_batch_shape = V.shape(), V_seq_shape = V.shape();
    V_batch_shape.RemoveDimRange(channel_axis, V.dims());
    V_seq_shape.RemoveDimRange(0, channel_axis+1);
    TensorShape O_batch_shape = O.shape(), O_seq_shape = O.shape();
    O_batch_shape.RemoveDimRange(channel_axis, O.dims());
    O_seq_shape.RemoveDimRange(0, channel_axis+1);
    TensorShape l_batch_shape = l.shape(), l_seq_shape = l.shape();
    l_batch_shape.RemoveDimRange(channel_axis, l.dims());
    l_seq_shape.RemoveDimRange(0, channel_axis);
    TensorShape m_batch_shape = m.shape(), m_seq_shape = m.shape();
    m_batch_shape.RemoveDimRange(channel_axis, m.dims());
    m_seq_shape.RemoveDimRange(0, channel_axis);
    TensorShape dO_batch_shape = dO.shape(), dO_seq_shape = dO.shape();
    dO_batch_shape.RemoveDimRange(channel_axis, dO.dims());
    dO_seq_shape.RemoveDimRange(0, channel_axis+1);



    OP_REQUIRES(context, Q_ch == K_ch,
      errors::InvalidArgument("The channel dimension of Q and K should be equal"));
    OP_REQUIRES(context, V_ch == O_ch,
      errors::InvalidArgument("The channel dimension of V and O should be equal"));


    OP_REQUIRES(context, Q_batch_shape == K_batch_shape && Q_batch_shape == V_batch_shape
                        && V_batch_shape == O_batch_shape && O_batch_shape == l_batch_shape
                        && l_batch_shape == m_batch_shape && m_batch_shape == dO_batch_shape,
      errors::InvalidArgument("The batch shape of all inputs should be equal, but Q_batch_shape = ", Q_batch_shape.DebugString(), ", K_batch_shape = ", K_batch_shape.DebugString(), ", V_batch_shape = ", V_batch_shape.DebugString(), ", O_batch_shape = ", O_batch_shape.DebugString(), ", l_batch_shape = ", l_batch_shape.DebugString(), ", m_batch_shape = ", m_batch_shape.DebugString(), ", dO_batch_shape = ", dO_batch_shape.DebugString(), " are received"));

    OP_REQUIRES(context, K_seq_shape == V_seq_shape,
      errors::InvalidArgument("The sequence shape of K and V should be equal, but K_seq_shape = ", K_seq_shape.DebugString(), ", V_seq_shape = ", V_seq_shape.DebugString(), " were received"));

    OP_REQUIRES(context, Q_seq_shape == O_seq_shape && O_seq_shape == l_seq_shape && l_seq_shape == m_seq_shape && m_seq_shape == dO_seq_shape,
      errors::InvalidArgument("The sequence shape of Q, O, l, m, and dO should be equal, but Q_seq_shape = ", Q_seq_shape.DebugString(), ", O_seq_shape = ", O_seq_shape.DebugString(), ", l_seq_shape = ", l_seq_shape.DebugString(), ", m_seq_shape = ", m_seq_shape.DebugString(), ", dO_seq_shape = ", dO_seq_shape.DebugString(), " were received"));

    const auto [reference_seq_shape, Q_seq_order_map, K_seq_order_map] = (*_sync_method)(Q_seq_shape, K_seq_shape);

    using MappedType = typename FP_TypeMapping<sizeof(T)>::FP_Type;
    FlashAttentionLauncher<MappedType,
                          std::remove_cv_t<decltype(reference_seq_shape)>,
                          std::remove_cv_t<decltype(Q_seq_order_map)>, AttentionPolicy> launcher{};

    using MappedL_Type = typename decltype(launcher)::L_T;

    Tensor* dQ = nullptr;
    Tensor* dK = nullptr;
    Tensor* dV = nullptr;
    Tensor Br_occupancy;

    TensorShape Br_occupancy_shape = Q_batch_shape;
    Br_occupancy_shape.AddDim(launcher.ComputeNumOfBrSections(Q_seq_shape.num_elements()));

    OP_REQUIRES_OK(context, context->allocate_output(0, Q.shape(), &dQ));

    OP_REQUIRES_OK(context, context->allocate_output(1, K.shape(), &dK));

    OP_REQUIRES_OK(context, context->allocate_output(2, V.shape(), &dV));

    OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT32, Br_occupancy_shape, &Br_occupancy));


    const MappedType* Q_data = Q.flat<MappedType>().data();
    const MappedType* K_data = K.flat<MappedType>().data();
    const MappedType* V_data = V.flat<MappedType>().data();
    const MappedType* O_data = O.flat<MappedType>().data();
    const MappedL_Type* l_data = l.flat<MappedL_Type>().data();
    const MappedType* m_data = m.flat<MappedType>().data();
    const MappedType* dO_data = dO.flat<MappedType>().data();

    MappedType* dQ_data = dQ->flat<MappedType>().data();
    MappedType* dK_data = dK->flat<MappedType>().data();
    MappedType* dV_data = dV->flat<MappedType>().data();

    uint32_t* Br_occupancy_data = Br_occupancy.flat<uint32_t>().data();

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    int shared_mem_per_block_optin = -1;
    int cuda_device = -1;
    OP_REQUIRES(context, cudaError_t::cudaSuccess == cudaGetDevice(&cuda_device),
                errors::Internal("Failed to get the current cuda device associated with this thread."));
    if (cudaError_t::cudaSuccess != cudaDeviceGetAttribute(&shared_mem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_device))
      shared_mem_per_block_optin = -1;


    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(dQ_data, 0, dQ->NumElements()*sizeof(MappedType), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor dQ"));

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(dK_data, 0, dK->NumElements()*sizeof(MappedType), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor dK"));

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(dV_data, 0, dV->NumElements()*sizeof(MappedType), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor dV"));

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(Br_occupancy_data, 0, Br_occupancy.NumElements()*sizeof(uint32_t), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor Br_occupancy"));


    cudaError_t launch_result = launcher.Backward(device.stream(), SharedMemoryDescriptor(device.sharedMemPerBlock(), shared_mem_per_block_optin),
                            Q_batch_shape.num_elements(),
                            Q_seq_shape.num_elements(),
                            K_seq_shape.num_elements(),
                            Q_ch, V_ch,
                            Q_data, K_data, V_data, O_data, l_data, m_data,
                            dO_data,
                            dQ_data, dK_data, dV_data,
                            Br_occupancy_data,
                            reference_seq_shape,
                            Q_seq_order_map, K_seq_order_map,
                            *_attention_policy);

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == launch_result,
                errors::Internal("Failed to launch the Backward kernel: ", cudaGetErrorString(launch_result), "(", launch_result, ")"));


  }

};

template <typename T, int SequenceDims>
class FullAttentionBackward : public FlashAttentionBackwardBase<T, SequenceDims, FullAttentionPolicy> {
 public:
  explicit FullAttentionBackward(OpKernelConstruction* cons)
  : FlashAttentionBackwardBase<T, SequenceDims, FullAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};


template <typename T, int SequenceDims>
class CausalAttentionBackward : public FlashAttentionBackwardBase<T, SequenceDims, CausalAttentionPolicy> {
 public:
  explicit CausalAttentionBackward(OpKernelConstruction* cons)
  : FlashAttentionBackwardBase<T, SequenceDims, CausalAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};

template <typename T, int SequenceDims>
class LocalAttentionBackward : public FlashAttentionBackwardBase<T, SequenceDims, LocalAttentionPolicy> {
 public:
  explicit LocalAttentionBackward(OpKernelConstruction* cons)
  : FlashAttentionBackwardBase<T, SequenceDims, LocalAttentionPolicy>(cons) {
    int window_size;
    int log2_stride_size;
    bool is_causal;

    OP_REQUIRES_OK(cons, cons->GetAttr("window_size", &window_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("log2_stride_size", &log2_stride_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("is_causal", &is_causal));

    this->_attention_policy.emplace(window_size, log2_stride_size, is_causal);
  }

};

#define REGISTER_BACKWARD_KERNELS(suffix, T, sequence_dims) \
  template class FullAttentionBackward<T, sequence_dims>; \
  template class CausalAttentionBackward<T, sequence_dims>; \
  template class LocalAttentionBackward<T, sequence_dims>; \
  REGISTER_KERNEL_BUILDER( \
      Name("FullAttentionBackward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      FullAttentionBackward<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("CausalAttentionBackward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      CausalAttentionBackward<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("LocalAttentionBackward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      LocalAttentionBackward<T, sequence_dims>);

ITERATE_TYPES_NO_HALF_SEQUENCE_DIMS(REGISTER_BACKWARD_KERNELS,)
ITERATE_SEQUENCE_DIMS(REGISTER_BACKWARD_KERNELS, Float16, Eigen::half)


#endif  // GOOGLE_CUDA

