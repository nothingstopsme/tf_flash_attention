

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


template <int SequenceDims>
absl::Status InferForwardOutputShapes(shape_inference::InferenceContext* c) {

  shape_inference::ShapeHandle Q_shape = c->input(0);
  shape_inference::ShapeHandle K_shape = c->input(1);
  shape_inference::ShapeHandle V_shape = c->input(2);

  // Simply deciding the shape of each output
  // without shape compatibility checks, which ares deferred to op's Compute()
  int Q_rank = c->Rank(Q_shape), K_rank = c->Rank(K_shape), V_rank = c->Rank(V_shape);

  if (Q_rank == K_rank && K_rank == V_rank && Q_rank >= SequenceDims+2) {
    int channel_axis = Q_rank - SequenceDims - 1;


    shape_inference::ShapeHandle O_shape, l_m_shape, temp_shape0, temp_shape1;

    absl::Status status = c->Subshape(V_shape, 0, channel_axis+1, &temp_shape0);
    if (!status.ok())
      return status;

    status = c->Subshape(Q_shape, channel_axis+1, &temp_shape1);
    if (!status.ok())
      return status;

    status = c->Concatenate(temp_shape0, temp_shape1, &O_shape);
    if (!status.ok())
      return status;

    status = c->Subshape(Q_shape, 0, channel_axis, &temp_shape0);
    if (!status.ok())
      return status;


    status = c->Subshape(Q_shape, channel_axis+1, &temp_shape1);
    if (!status.ok())
      return status;


    status = c->Concatenate(temp_shape0, temp_shape1, &l_m_shape);
    if (!status.ok())
      return status;


    c->set_output(0, O_shape);
    c->set_output(1, l_m_shape);
    c->set_output(2, l_m_shape);

    return OkStatus();
  }
  else
    return absl::InvalidArgumentError("Failed to infer the shape of outputs as the shape of some inputs might be incorrect");

}

inline absl::Status InferFlopsShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return OkStatus();
}

template <int SequenceDims>
inline auto VerifyAndExtractShapes(const TensorShape& Q_shape, const TensorShape& K_shape, const TensorShape& V_shape) {

  if (Q_shape.dims() != K_shape.dims() || K_shape.dims() != V_shape.dims())
    throw errors::InvalidArgument("The number of dimensions of Q, K, and V should be equal");

  if (Q_shape.dims() < SequenceDims+2)
    throw errors::InvalidArgument("The number of dimensions of Q, K, and V should be >= ", SequenceDims+2);

  int channel_axis = Q_shape.dims() - SequenceDims - 1;


  int64_t Q_ch = Q_shape.dim_size(channel_axis);
  int64_t K_ch = K_shape.dim_size(channel_axis);
  int64_t V_ch = V_shape.dim_size(channel_axis);


  TensorShape Q_batch_shape = Q_shape,  Q_seq_shape = Q_shape;
  Q_batch_shape.RemoveDimRange(channel_axis, Q_shape.dims());
  Q_seq_shape.RemoveDimRange(0, channel_axis+1);
  TensorShape K_batch_shape = K_shape, K_seq_shape = K_shape;
  K_batch_shape.RemoveDimRange(channel_axis, K_shape.dims());
  K_seq_shape.RemoveDimRange(0, channel_axis+1);
  TensorShape V_batch_shape = V_shape, V_seq_shape = V_shape;
  V_batch_shape.RemoveDimRange(channel_axis, V_shape.dims());
  V_seq_shape.RemoveDimRange(0, channel_axis+1);



  if (Q_ch != K_ch)
    throw errors::InvalidArgument("The channel dimension of Q and K should be equal");

  if (Q_batch_shape != K_batch_shape || Q_batch_shape != V_batch_shape)
    throw errors::InvalidArgument("The batch shape of all inputs should be equal, but Q_batch_shape = ", Q_batch_shape.DebugString(), ", K_batch_shape = ", K_batch_shape.DebugString(), ", V_batch_shape = ", V_batch_shape.DebugString(), " were received");

  if (K_seq_shape != V_seq_shape)
    throw errors::InvalidArgument("The sequence shape of K and V are expected to be equal, but K_seq_shape = ", K_seq_shape.DebugString(), ", V_seq_shape = ", V_seq_shape.DebugString(), " are detected");


  return std::make_tuple(Q_batch_shape, Q_seq_shape, Q_ch,
                          K_batch_shape, K_seq_shape, K_ch,
                          V_batch_shape, V_seq_shape, V_ch);

}

} // anonymous namespace

#define REGISTER_TF_OPS(unused, sequence_dims) \
  REGISTER_OP("FullAttentionForward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Output("o: T") \
    .Output("l: float") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("CausalAttentionForward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Output("o: T") \
    .Output("l: float") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("LocalAttentionForward" #sequence_dims "dFloat16") \
    .Attr("T: {float16}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Attr("window_size: int >= 1") \
    .Attr("log2_stride_size: int >= 0") \
    .Attr("is_causal: bool") \
    .Output("o: T") \
    .Output("l: float") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("FullAttentionForward" #sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Output("o: T") \
    .Output("l: T") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("CausalAttentionForward" #sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Output("o: T") \
    .Output("l: T") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("LocalAttentionForward" #sequence_dims "d") \
    .Attr("T: {float, double}") \
    .Input("q: T") \
    .Input("k: T") \
    .Input("v: T") \
    .Attr("sync_mode: string") \
    .Attr("window_size: int >= 1") \
    .Attr("log2_stride_size: int >= 0") \
    .Attr("is_causal: bool") \
    .Output("o: T") \
    .Output("l: T") \
    .Output("m: T") \
    .SetShapeFn(&InferForwardOutputShapes<sequence_dims>); \
  \
  REGISTER_OP("EstimateFullAttentionForward" #sequence_dims "dFlops") \
    .Attr("q_shape: shape") \
    .Attr("k_shape: shape") \
    .Attr("v_shape: shape") \
    .Attr("dtype: {float16, float, double}") \
    .Attr("sync_mode: string") \
    .Output("flops: float") \
    .SetShapeFn(&InferFlopsShape); \
  \
  REGISTER_OP("EstimateCausalAttentionForward" #sequence_dims "dFlops") \
    .Attr("q_shape: shape") \
    .Attr("k_shape: shape") \
    .Attr("v_shape: shape") \
    .Attr("dtype: {float16, float, double}") \
    .Attr("sync_mode: string") \
    .Output("flops: float") \
    .SetShapeFn(&InferFlopsShape); \
  \
  REGISTER_OP("EstimateLocalAttentionForward" #sequence_dims "dFlops") \
    .Attr("q_shape: shape") \
    .Attr("k_shape: shape") \
    .Attr("v_shape: shape") \
    .Attr("dtype: {float16, float, double}") \
    .Attr("sync_mode: string") \
    .Attr("window_size: int >= 1") \
    .Attr("log2_stride_size: int >= 0") \
    .Attr("is_causal: bool") \
    .Output("flops: float") \
    .SetShapeFn(&InferFlopsShape);

// What ITERATE_SEQUENCE_DIMS() does internally is to take
// its first input as the macro function to be expanded,
// and passing a list of arguments containing the rest of inputs appended
// with a dimension value. Therefore in this case REGISTER_TF_OPS() will receive
// two arguments: the first one is empty (expanded to nothing),
// while the second one is that dimension value
ITERATE_SEQUENCE_DIMS(REGISTER_TF_OPS,)

template <typename T, int SequenceDims, typename AttentionPolicy>
class FlashAttentionForwardBase : public OpKernel {
 private:
  std::optional<SyncMethod<SequenceDims>> _sync_method;

 protected:
  std::optional<AttentionPolicy> _attention_policy;

 public:
  explicit FlashAttentionForwardBase(OpKernelConstruction* cons, const char* fixed_sync_mode = nullptr)
  : OpKernel(cons) {
    static_assert(SequenceDims >= 1, "SequenceDims should be >= 1");

    std::string sync_mode;
    if (fixed_sync_mode)
      sync_mode.assign(fixed_sync_mode);
    else
      OP_REQUIRES_OK(cons, cons->GetAttr("sync_mode", &sync_mode));

    SyncMethods::Lookup<SequenceDims>(sync_mode, _sync_method);
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

    TensorShape Q_batch_shape, Q_seq_shape;
    TensorShape K_batch_shape, K_seq_shape;
    TensorShape V_batch_shape, V_seq_shape;
    int64_t Q_ch, K_ch, V_ch;

    try {
      std::tie(Q_batch_shape, Q_seq_shape, Q_ch,
                K_batch_shape, K_seq_shape, K_ch,
                V_batch_shape, V_seq_shape, V_ch) = VerifyAndExtractShapes<SequenceDims>(Q.shape(), K.shape(), V.shape());
    } catch (const absl::Status& status) {

      OP_REQUIRES_OK(context, status);
    }

    const auto [reference_seq_shape, Q_seq_order_map, K_seq_order_map] = (*_sync_method)(Q_seq_shape, K_seq_shape);

    using MappedType = typename FP_TypeMapping<sizeof(T)>::FP_Type;
    FlashAttentionLauncher<MappedType,
                            std::remove_cv_t<decltype(reference_seq_shape)>,
                            std::remove_cv_t<decltype(Q_seq_order_map)>, AttentionPolicy> launcher{};

    using MappedL_Type = typename decltype(launcher)::L_T;


    Tensor* O = nullptr;
    Tensor* l = nullptr;
    Tensor* m = nullptr;
    Tensor Br_occupancy;

    TensorShape O_shape = Q_batch_shape, l_m_shape = Q_batch_shape, Br_occupancy_shape = Q_batch_shape;
    O_shape.AddDim(V_ch);
    O_shape.AppendShape(Q_seq_shape);
    l_m_shape.AppendShape(Q_seq_shape);
    Br_occupancy_shape.AddDim(launcher.ComputeNumOfBrSections(Q_seq_shape.num_elements()));

    OP_REQUIRES_OK(context, context->allocate_output(0, O_shape, &O));

    OP_REQUIRES_OK(context, context->allocate_output(1, l_m_shape, &l));

    OP_REQUIRES_OK(context, context->allocate_output(2, l_m_shape, &m));

    OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT32, Br_occupancy_shape, &Br_occupancy));


    const MappedType* Q_data = Q.flat<MappedType>().data();
    const MappedType* K_data = K.flat<MappedType>().data();
    const MappedType* V_data = V.flat<MappedType>().data();

    MappedType* O_data = O->flat<MappedType>().data();
    MappedL_Type* l_data = l->flat<MappedL_Type>().data();
    MappedType* m_data = m->flat<MappedType>().data();

    uint32_t* Br_occupancy_data = Br_occupancy.flat<uint32_t>().data();

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    int shared_mem_per_block_optin = -1;
    int cuda_device = -1;
    OP_REQUIRES(context, cudaError_t::cudaSuccess == cudaGetDevice(&cuda_device),
                errors::Internal("Failed to get the current cuda device associated with this thread."));
    if (cudaError_t::cudaSuccess != cudaDeviceGetAttribute(&shared_mem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_device))
      shared_mem_per_block_optin = -1;


    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(O_data, 0, O->NumElements()*sizeof(MappedType), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor O"));

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(l_data, 0, l->NumElements()*sizeof(MappedL_Type), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor l"));

    // By initialising every byte to 0xfa,
    // we effectively set every element in m_data, which is of the type MappedType,
    // to a negative number far away from 0
    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(m_data, 0xfa, m->NumElements()*sizeof(MappedType), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to initailise the tensor m"));

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == cudaMemsetAsync(Br_occupancy_data, 0, Br_occupancy.NumElements()*sizeof(uint32_t), device.stream()),
                errors::Internal("Failed to conduct cudaMemset to zero out the tensor Br_occupancy"));

    cudaError_t launch_result = launcher.Forward(device.stream(), SharedMemoryDescriptor(device.sharedMemPerBlock(), shared_mem_per_block_optin),
                            Q_batch_shape.num_elements(),
                            Q_seq_shape.num_elements(),
                            K_seq_shape.num_elements(),
                            Q_ch, V_ch,
                            Q_data, K_data, V_data,
                            O_data, l_data, m_data,
                            Br_occupancy_data,
                            reference_seq_shape,
                            Q_seq_order_map, K_seq_order_map,
                            *_attention_policy);

    OP_REQUIRES(context,
                cudaError_t::cudaSuccess == launch_result,
                errors::Internal("Failed to launch the Forward kernel: ", cudaGetErrorString(launch_result), "(", launch_result, ")"));
  }

};

template <typename T, int SequenceDims, typename AttentionPolicy>
class FlashAttentionForwardFlopsEstimationBase : public OpKernel {
 private:
  TensorShape _Q_shape, _K_shape, _V_shape;
  std::optional<SyncMethod<SequenceDims>> _sync_method;

 protected:
  std::optional<AttentionPolicy> _attention_policy;

 public:
  explicit FlashAttentionForwardFlopsEstimationBase(OpKernelConstruction* cons, const char* fixed_sync_mode = nullptr)
  : OpKernel(cons) {
    static_assert(SequenceDims >= 1, "SequenceDims should be >= 1");
    std::string sync_mode;

    OP_REQUIRES_OK(cons, cons->GetAttr("q_shape", &_Q_shape));
    OP_REQUIRES_OK(cons, cons->GetAttr("k_shape", &_K_shape));
    OP_REQUIRES_OK(cons, cons->GetAttr("v_shape", &_V_shape));

    if (fixed_sync_mode)
      sync_mode.assign(fixed_sync_mode);
    else
      OP_REQUIRES_OK(cons, cons->GetAttr("sync_mode", &sync_mode));

    SyncMethods::Lookup<SequenceDims>(sync_mode, _sync_method);
    OP_REQUIRES(cons, _sync_method.has_value(),
                errors::InvalidArgument("Unsupported sync_mode: ", sync_mode));

  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context,
                _attention_policy.has_value(),
                errors::Internal("attention_policy has set to be set"));

    TensorShape Q_batch_shape, Q_seq_shape;
    TensorShape K_batch_shape, K_seq_shape;
    TensorShape V_batch_shape, V_seq_shape;
    int64_t Q_ch, K_ch, V_ch;

    try {
      std::tie(Q_batch_shape, Q_seq_shape, Q_ch,
                K_batch_shape, K_seq_shape, K_ch,
                V_batch_shape, V_seq_shape, V_ch) = VerifyAndExtractShapes<SequenceDims>(_Q_shape, _K_shape, _V_shape);
    } catch (const absl::Status& status) {

      OP_REQUIRES_OK(context, status);
    }

    const auto [reference_seq_shape, Q_seq_order_map, K_seq_order_map] = (*_sync_method)(Q_seq_shape, K_seq_shape);


    using MappedType = typename FP_TypeMapping<sizeof(T)>::FP_Type;
    FlashAttentionLauncher<MappedType,
                            std::remove_cv_t<decltype(reference_seq_shape)>,
                            std::remove_cv_t<decltype(Q_seq_order_map)>, AttentionPolicy> launcher{};

    Tensor* flops = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &flops));

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    int shared_mem_per_block_optin = -1;
    int cuda_device = -1;
    OP_REQUIRES(context, cudaError_t::cudaSuccess == cudaGetDevice(&cuda_device),
                errors::Internal("Failed to get the current cuda device associated with this thread."));
    if (cudaError_t::cudaSuccess != cudaDeviceGetAttribute(&shared_mem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_device))
      shared_mem_per_block_optin = -1;

    // The variable "flops" must reside in CPU memory due to the setup given on op registration,
    // so it is OK to directly update it by calling EstimateForwardFlops(),
    // which is also run on CPU
    launcher.EstimateForwardFlops(SharedMemoryDescriptor(device.sharedMemPerBlock(), shared_mem_per_block_optin),
                                                            Q_batch_shape.num_elements(),
                                                            Q_seq_shape.num_elements(),
                                                            K_seq_shape.num_elements(),
                                                            Q_ch, V_ch,
                                                            reference_seq_shape,
                                                            Q_seq_order_map, K_seq_order_map,
                                                            *_attention_policy,
                                                            flops->flat<float>()(0));



  }
};

template <typename T, int SequenceDims>
class FullAttentionForward : public FlashAttentionForwardBase<T, SequenceDims, FullAttentionPolicy> {
 public:
  explicit FullAttentionForward(OpKernelConstruction* cons)
  : FlashAttentionForwardBase<T, SequenceDims, FullAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};

template <typename T, int SequenceDims>
class CausalAttentionForward : public FlashAttentionForwardBase<T, SequenceDims, CausalAttentionPolicy> {
 public:
  explicit CausalAttentionForward(OpKernelConstruction* cons)
  : FlashAttentionForwardBase<T, SequenceDims, CausalAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};

template <typename T, int SequenceDims>
class LocalAttentionForward : public FlashAttentionForwardBase<T, SequenceDims, LocalAttentionPolicy> {
 public:
  explicit LocalAttentionForward(OpKernelConstruction* cons)
  : FlashAttentionForwardBase<T, SequenceDims, LocalAttentionPolicy>(cons) {
    int window_size;
    int log2_stride_size;
    bool is_causal;

    OP_REQUIRES_OK(cons, cons->GetAttr("window_size", &window_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("log2_stride_size", &log2_stride_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("is_causal", &is_causal));

    this->_attention_policy.emplace(window_size, log2_stride_size, is_causal);
  }

};

template <typename T, int SequenceDims>
class FullAttentionForwardFlopsEstimation : public FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, FullAttentionPolicy> {
 public:
  explicit FullAttentionForwardFlopsEstimation(OpKernelConstruction* cons)
  : FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, FullAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};

template <typename T, int SequenceDims>
class CausalAttentionForwardFlopsEstimation : public FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, CausalAttentionPolicy> {
 public:
  explicit CausalAttentionForwardFlopsEstimation(OpKernelConstruction* cons)
  : FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, CausalAttentionPolicy>(cons) {
    this->_attention_policy.emplace();
  }
};

template <typename T, int SequenceDims>
class LocalAttentionForwardFlopsEstimation : public FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, LocalAttentionPolicy> {
 public:
  explicit LocalAttentionForwardFlopsEstimation(OpKernelConstruction* cons)
  : FlashAttentionForwardFlopsEstimationBase<T, SequenceDims, LocalAttentionPolicy>(cons) {
    int window_size;
    int log2_stride_size;
    bool is_causal;

    OP_REQUIRES_OK(cons, cons->GetAttr("window_size", &window_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("log2_stride_size", &log2_stride_size));
    OP_REQUIRES_OK(cons, cons->GetAttr("is_causal", &is_causal));

    this->_attention_policy.emplace(window_size, log2_stride_size, is_causal);
  }
};


#define REGISTER_FORWARD_KERNELS(suffix, T, sequence_dims) \
  template class FullAttentionForward<T, sequence_dims>; \
  template class CausalAttentionForward<T, sequence_dims>; \
  template class LocalAttentionForward<T, sequence_dims>; \
  REGISTER_KERNEL_BUILDER( \
      Name("FullAttentionForward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      FullAttentionForward<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("CausalAttentionForward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      CausalAttentionForward<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("LocalAttentionForward" #sequence_dims "d" #suffix).Device(DEVICE_GPU) \
      .TypeConstraint<T>("T"), \
      LocalAttentionForward<T, sequence_dims>);

#define REGISTER_FLOPS_ESTIMATORS(T, sequence_dims) \
  template class FullAttentionForwardFlopsEstimation<T, sequence_dims>; \
  template class CausalAttentionForwardFlopsEstimation<T, sequence_dims>; \
  template class LocalAttentionForwardFlopsEstimation<T, sequence_dims>; \
  REGISTER_KERNEL_BUILDER( \
      Name("EstimateFullAttentionForward" #sequence_dims "dFlops").Device(DEVICE_GPU) \
      .TypeConstraint<T>("dtype") \
      .HostMemory("flops"), \
      FullAttentionForwardFlopsEstimation<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("EstimateCausalAttentionForward" #sequence_dims "dFlops").Device(DEVICE_GPU) \
      .TypeConstraint<T>("dtype") \
      .HostMemory("flops"), \
      CausalAttentionForwardFlopsEstimation<T, sequence_dims>); \
  REGISTER_KERNEL_BUILDER( \
      Name("EstimateLocalAttentionForward" #sequence_dims "dFlops").Device(DEVICE_GPU) \
      .TypeConstraint<T>("dtype") \
      .HostMemory("flops"), \
      LocalAttentionForwardFlopsEstimation<T, sequence_dims>);




ITERATE_TYPES_NO_HALF_SEQUENCE_DIMS(REGISTER_FORWARD_KERNELS,)
ITERATE_SEQUENCE_DIMS(REGISTER_FORWARD_KERNELS, Float16, Eigen::half)

ITERATE_TYPES_SEQUENCE_DIMS(REGISTER_FLOPS_ESTIMATORS)

#endif  // GOOGLE_CUDA
