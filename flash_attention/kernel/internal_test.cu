#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "flash_attention.h"
#include "type_util.h"
#include "sync_methods.h"
#include <sstream>

namespace {

using namespace cute;

using namespace cuda_launch;

template <typename T, typename = void>
struct Precision {
  static constexpr float value = 1e-9f;
};

template <typename T>
struct Precision<T, typename std::enable_if<std::is_same<T, half>::value>::type> {
  static constexpr float value = 1e-2f;
};

template <typename T>
struct Precision<T, typename std::enable_if<std::is_same<T, float>::value>::type> {
  static constexpr float value = 1e-6f;
};


class Stopwatch
{
 public:
  Stopwatch(const cudaStream_t stream_handle)
    : stream_handle_(stream_handle) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~Stopwatch() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  Stopwatch& operator=(const Stopwatch&) = delete;

  void start() const {
    cudaEventRecord(start_, stream_handle_);
  }

  void stop() const {
    cudaEventRecord(stop_, stream_handle_);
  }


  float seconds() const {
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3f;
  }

 private:
  const cudaStream_t stream_handle_;
  cudaEvent_t start_, stop_;
};

#if 0
template <typename T>
CUTE_HOST_DEVICE
static constexpr auto AddSquared(const T& a, const T& b) {
  return a + b * b;
}

template <typename DataType>
struct AlibiMinusAndNormalise {
  DataType normaliser;

  template <typename InputType>
  CUTE_HOST_DEVICE
  constexpr DataType operator()(const InputType& a, const InputType& b) const {
    return static_cast<DataType>(a-b) / normaliser;
  }
};
#endif

template<typename DataType>
void TestForward() {

  constexpr int B_SIZE = 1;
  constexpr int H_SIZE = 8;
  constexpr int Q_SIZE = 1024;
  constexpr int K_SIZE = 1024;
  constexpr int D_SIZE = 32;
  constexpr int V_D_SIZE = 32;
  //constexpr int WINDOW_SIZE = 4;

  constexpr auto REFERENCE_SIZE = std::max(Q_SIZE, K_SIZE);

  SequenceDescriptorPack Q_K_seq_desc_pack(1);
  Q_K_seq_desc_pack.reference_shape.push_back(REFERENCE_SIZE);
  Q_K_seq_desc_pack.Q_desc.shape.push_back(Q_SIZE);
  Q_K_seq_desc_pack.Q_desc.stride.push_back(REFERENCE_SIZE / Q_SIZE);
  Q_K_seq_desc_pack.Q_desc.offset.push_back(0);
  Q_K_seq_desc_pack.K_desc.shape.push_back(K_SIZE);
  Q_K_seq_desc_pack.K_desc.stride.push_back(REFERENCE_SIZE / K_SIZE);
  Q_K_seq_desc_pack.K_desc.offset.push_back(0);

  const auto [reference_seq_shape, Q_seq_order_map, K_seq_order_map] = SyncMethod<1>::GenerateOrderMap(Q_K_seq_desc_pack);
  #if 0
  const AlibiMinusAndNormalise<DataType> alibi_normaliser{static_cast<DataType>(size(reference_shape))};
  const auto reference_with_heads_shape = append(reference_shape, H_SIZE);
  #endif

  FullAttentionPolicy attention_policy;
  FlashAttentionLauncher<DataType,
                          std::remove_cv_t<decltype(reference_seq_shape)>,
                          std::remove_cv_t<decltype(Q_seq_order_map)>, decltype(attention_policy)> launcher{};
  using L_DataType = typename decltype(launcher)::L_T;


  thrust::host_vector<DataType> Q(B_SIZE*H_SIZE*Q_SIZE*D_SIZE);
  thrust::host_vector<DataType> K(B_SIZE*H_SIZE*K_SIZE*D_SIZE);
  thrust::host_vector<DataType> V(B_SIZE*H_SIZE*K_SIZE*V_D_SIZE);
  thrust::host_vector<DataType> O(B_SIZE*H_SIZE*Q_SIZE*V_D_SIZE, DataType(0));
  thrust::host_vector<L_DataType> l(B_SIZE*H_SIZE*Q_SIZE, L_DataType(0));
  thrust::host_vector<DataType> m(B_SIZE*H_SIZE*Q_SIZE, TypeUtil<DataType>::GetNegInfApprox());

  thrust::host_vector<DataType> ans_O = O;
  thrust::host_vector<L_DataType> ans_l = l;
  thrust::host_vector<DataType> ans_m = m;

  const auto dot_scaler = TypeUtil<DataType>::RSqrt(static_cast<DataType>(D_SIZE));
  const auto masking_value = TypeUtil<DataType>::GetNegInfApprox();

  // Doing a naive CPU-version of forward
  for (int b_index = 0; b_index < B_SIZE; ++b_index) {
    for (int h_index = 0; h_index < H_SIZE; ++h_index) {
      // Initialising Q
      for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
        for (int d_index = 0; d_index < D_SIZE; ++d_index) {
          DataType random_q = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
          Q[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] = random_q;
        }
      }
      // Initialising K and V
      for (int k_index = 0; k_index < K_SIZE; ++k_index) {
        for (int d_index = 0; d_index < max(D_SIZE, V_D_SIZE); ++d_index) {
          if (d_index < D_SIZE) {
            DataType random_k = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            K[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index] = random_k;
          }

          if (d_index < V_D_SIZE) {
            DataType random_v = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            V[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index] = random_v;
          }
        }
      }

      // Computing S = Q @ K^T
      thrust::host_vector<DataType> P_S(Q_SIZE*K_SIZE);
      for (int d_index = 0; d_index < D_SIZE; ++d_index) {
        for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
          for (int k_index = 0; k_index < K_SIZE; ++k_index) {
            if (d_index == 0)
              P_S[q_index*K_SIZE + k_index] = DataType(0);

            P_S[q_index*K_SIZE + k_index] += Q[b_index*H_SIZE*Q_SIZE*D_SIZE+ h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] * K[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index];

          }
        }
      }

      //DataType alibi_slope = TypeUtil<DataType>::Exp(static_cast<DataType>(h_index+1) * TypeUtil<DataType>::Max(DataType(-8) / static_cast<DataType>(H_SIZE), DataType(-1))) / dot_scaler;

      for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
        DataType max(masking_value);
        L_DataType sum(0);

        // Finding the maximum value of this row
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          DataType& current = P_S[q_index*K_SIZE + k_index];
          current *= dot_scaler;
          #if 0
          if constexpr (ALIBI_ON) {
            const auto q_order = get<0>(Q_seq_order_map(q_index));
            const auto k_order = get<0>(K_seq_order_map(k_index));

            const auto q_coord = idx2crd(q_order, reference_shape);
            const auto k_coord = idx2crd(k_order, reference_shape);

            const auto dist = TypeUtil<DataType>::Sqrt(fold(transform(q_coord, k_coord, alibi_normaliser), DataType(0), AddSquared<DataType>)) * alibi_normaliser.normaliser;

            current -= alibi_slope * dist;
          }
          #endif

          #if 0
          if (q_index >= k_index)
            max = (max < current) ? current : max;
          else
            current = TypeUtil<DataType>::GetNegInfApprox();
          #else
          max = (max < current) ? current : max;
          #endif
        }

        ans_m[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] = max;

        // Preparing for softmax by updating entry values and summing them up
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          DataType& current = P_S[q_index*K_SIZE + k_index];
          if (current <= masking_value)
            current = DataType(0);
          else
            current = TypeUtil<DataType>::Exp((current - max));
          sum += static_cast<L_DataType>(current);
        }

        ans_l[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] = sum;

        // Doing normalisation to turn S into P containing a proper distribution,
        // and computing O = P @ V
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          const auto p = P_S[q_index*K_SIZE + k_index] = static_cast<DataType>(static_cast<L_DataType>(P_S[q_index*K_SIZE + k_index]) / sum);

          for (int d_index = 0; d_index < V_D_SIZE; ++d_index) {
            ans_O[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index] += p * V[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index];
          }
        }
      }
    }
  }

  // Doing the GPU-version of forward
  thrust::device_vector<DataType> d_Q = Q;
  thrust::device_vector<DataType> d_K = K;
  thrust::device_vector<DataType> d_V = V;
  thrust::device_vector<DataType> d_O = O;
  thrust::device_vector<L_DataType> d_l = l;
  thrust::device_vector<DataType> d_m = m;

  thrust::device_vector<uint32_t> Br_occupancy(B_SIZE*H_SIZE*launcher.ComputeNumOfBrSections(Q_SIZE), 0);


  cudaStream_t stream_handle = nullptr;
  Stopwatch stopwatch(stream_handle);

  stopwatch.start();
  cudaError_t launch_result = launcher.Forward(
        stream_handle, SharedMemoryDescriptor(48<<10, 48<<10),
        B_SIZE*H_SIZE,
        Q_SIZE, K_SIZE,
        D_SIZE, V_D_SIZE,
        d_Q.data().get(),
        d_K.data().get(),
        d_V.data().get(),
        d_O.data().get(),
        d_l.data().get(),
        d_m.data().get(),
        Br_occupancy.data().get(),
        reference_seq_shape,
        Q_seq_order_map, K_seq_order_map,
        attention_policy);
  stopwatch.stop();

  if (cudaError_t::cudaSuccess != launch_result) {
    std::stringstream message;
    message << "Failed to launch the Forward kernel: "
            << cudaGetErrorString(launch_result)
            << "(" << launch_result << ")";
    throw std::runtime_error(message.str());
  }

  // Calling stopwatch.seconds() will enforce the synchronisation between the host and the device,
  // so no need to invoke cudaDeviceSynchronize()
  //cudaDeviceSynchronize();
  float elapsed_seconds = stopwatch.seconds();
  printf("kernel running time: %f secs\n", elapsed_seconds);

  O = d_O;
  l = d_l;
  m = d_m;

  // Comparing O, l, and m against ans_O, ans_l, and ans_m
  //
  // To account for precision errors which might get enlarged through accumulation,
  // normalisation by the number of entries from which the result is computed is conducted
  float error_count_O = 0.0, error_count_l = 0.0, error_count_m = 0.0;
  float max_error_O = 0.0, max_error_l = 0.0, max_error_m = 0.0;
  for (int b_index = 0; b_index < B_SIZE; ++b_index) {
    for (int h_index = 0; h_index < H_SIZE; ++h_index) {
      for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
        const auto error_l = std::abs(static_cast<float>(l[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] - ans_l[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index])) / K_SIZE;
        const auto error_m = std::abs(static_cast<float>(m[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] - ans_m[b_index*H_SIZE*Q_SIZE+ h_index*Q_SIZE + q_index])) / D_SIZE;
        error_count_l += static_cast<float>(error_l > Precision<DataType>::value);
        error_count_m += static_cast<float>(error_m > Precision<DataType>::value);

        max_error_l = max(error_l, max_error_l);
        max_error_m = max(error_m, max_error_m);
        for (int d_index = 0; d_index < V_D_SIZE; ++d_index) {
          const auto error_O = std::abs(static_cast<float>(O[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index] - ans_O[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index])) / K_SIZE;
          error_count_O += static_cast<float>(error_O > Precision<DataType>::value);

          max_error_O = max(error_O, max_error_O);
        }
      }
    }
  }
  printf("error rate: O = %f, l = %f, m = %f\n",
          error_count_O/O.size(),
          error_count_l/l.size(),
          error_count_m/m.size());
  printf("max error: O = %f, l = %f, m = %f\n",
          max_error_O,
          max_error_l,
          max_error_m);

}


template<typename DataType>
void TestBackward() {


  constexpr int B_SIZE = 1;
  constexpr int H_SIZE = 8;
  constexpr int Q_SIZE = 1024;
  constexpr int K_SIZE = 1024;
  constexpr int D_SIZE = 32;
  constexpr int V_D_SIZE = 32;
  //constexpr int WINDOW_SIZE = 4;

  constexpr auto REFERENCE_SIZE = std::max(Q_SIZE, K_SIZE);

  SequenceDescriptorPack Q_K_seq_desc_pack(1);
  Q_K_seq_desc_pack.reference_shape.push_back(REFERENCE_SIZE);
  Q_K_seq_desc_pack.Q_desc.shape.push_back(Q_SIZE);
  Q_K_seq_desc_pack.Q_desc.stride.push_back(REFERENCE_SIZE / Q_SIZE);
  Q_K_seq_desc_pack.Q_desc.offset.push_back(0);
  Q_K_seq_desc_pack.K_desc.shape.push_back(K_SIZE);
  Q_K_seq_desc_pack.K_desc.stride.push_back(REFERENCE_SIZE / K_SIZE);
  Q_K_seq_desc_pack.K_desc.offset.push_back(0);

  const auto [reference_seq_shape, Q_seq_order_map, K_seq_order_map] = SyncMethod<1>::GenerateOrderMap(Q_K_seq_desc_pack);
  #if 0
  const AlibiMinusAndNormalise<DataType> alibi_normaliser{static_cast<DataType>(size(reference_shape))};
  const auto reference_with_heads_shape = append(reference_shape, H_SIZE);
  #endif

  FullAttentionPolicy attention_policy;
  FlashAttentionLauncher<DataType,
                          std::remove_cv_t<decltype(reference_seq_shape)>,
                          std::remove_cv_t<decltype(Q_seq_order_map)>, decltype(attention_policy)> launcher{};


  using L_DataType = typename decltype(launcher)::L_T;


  thrust::host_vector<DataType> Q(B_SIZE*H_SIZE*Q_SIZE*D_SIZE);
  thrust::host_vector<DataType> dQ(B_SIZE*H_SIZE*Q_SIZE*D_SIZE, DataType(0));
  thrust::host_vector<DataType> K(B_SIZE*H_SIZE*K_SIZE*D_SIZE);
  thrust::host_vector<DataType> dK(B_SIZE*H_SIZE*K_SIZE*D_SIZE, DataType(0));
  thrust::host_vector<DataType> V(B_SIZE*H_SIZE*K_SIZE*V_D_SIZE);
  thrust::host_vector<DataType> dV(B_SIZE*H_SIZE*K_SIZE*V_D_SIZE, DataType(0));
  thrust::host_vector<DataType> O(B_SIZE*H_SIZE*Q_SIZE*V_D_SIZE);
  thrust::host_vector<DataType> dO(B_SIZE*H_SIZE*Q_SIZE*V_D_SIZE);
  thrust::host_vector<L_DataType> l(B_SIZE*H_SIZE*Q_SIZE);
  thrust::host_vector<DataType> m(B_SIZE*H_SIZE*Q_SIZE);

  thrust::host_vector<DataType> P_S(Q_SIZE*K_SIZE), d_P_S(Q_SIZE*K_SIZE);

  thrust::host_vector<DataType> ans_dQ = dQ;
  thrust::host_vector<DataType> ans_dK = dK;
  thrust::host_vector<DataType> ans_dV = dV;


  const auto dot_scaler = TypeUtil<DataType>::RSqrt(static_cast<DataType>(D_SIZE));
  const auto masking_value = TypeUtil<DataType>::GetNegInfApprox();

  // Doing a naive CPU-version of backward
  for (int b_index = 0; b_index < B_SIZE; ++b_index) {
    for (int h_index = 0; h_index < H_SIZE; ++h_index) {
      // Initialising Q and dO
      for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
        for (int d_index = 0; d_index < max(D_SIZE, V_D_SIZE); ++d_index) {
          if (d_index < D_SIZE) {
            DataType random_q = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            Q[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] = random_q;
          }

          if (d_index < V_D_SIZE) {
            DataType random_d_o = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            dO[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index] = random_d_o;
          }
        }
      }
      // Initialising K and V
      for (int k_index = 0; k_index < K_SIZE; ++k_index) {
        for (int d_index = 0; d_index < max(D_SIZE, V_D_SIZE); ++d_index) {
          if (d_index < D_SIZE) {
            DataType random_k = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            K[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index] = random_k;
          }

          if (d_index < V_D_SIZE) {
            DataType random_v = static_cast<DataType>(4*(static_cast<double>(rand()) / RAND_MAX) - 2);
            V[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index] = random_v;
          }
        }
      }

      // Computing S = Q @ K^T and dP = dO @ V^T
      for (int d_index = 0; d_index < max(D_SIZE, V_D_SIZE); ++d_index) {
        for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
          for (int k_index = 0; k_index < K_SIZE; ++k_index) {
            if (d_index == 0) {
              P_S[q_index*K_SIZE + k_index] = DataType(0);
              d_P_S[q_index*K_SIZE + k_index] = DataType(0);
            }

            if (d_index < D_SIZE) {
              P_S[q_index*K_SIZE + k_index] += Q[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] * K[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index];
            }

            if (d_index < V_D_SIZE) {
              d_P_S[q_index*K_SIZE + k_index] += dO[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] * V[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index];
            }
          }
        }
      }

      //DataType alibi_slope = TypeUtil<DataType>::Exp(static_cast<DataType>(h_index+1) * TypeUtil<DataType>::Max(DataType(-8) / static_cast<DataType>(H_SIZE), DataType(-1))) / dot_scaler;
      for (int q_index = 0; q_index < Q_SIZE; ++q_index) {
        DataType max(masking_value);
        L_DataType sum(0);

        // Finding the maximum value of this row
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          DataType& current = P_S[q_index*K_SIZE + k_index];
          current *= dot_scaler;

          #if 0
          if constexpr (ALIBI_ON) {
            const auto q_order = get<0>(Q_seq_order_map(q_index));
            const auto k_order = get<0>(K_seq_order_map(k_index));

            const auto q_coord = idx2crd(q_order, reference_shape);
            const auto k_coord = idx2crd(k_order, reference_shape);

            const auto dist = TypeUtil<DataType>::Sqrt(fold(transform(q_coord, k_coord, alibi_normaliser), DataType(0), AddSquared<DataType>)) * alibi_normaliser.normaliser;
            current -= alibi_slope * dist;
          }
          #endif

          #if 0
          if (q_index >= k_index)
            max = (max < current) ? current : max;
          else
            current = TypeUtil<DataType>::GetNegInfApprox();
          #else
          max = (max < current) ? current : max;
          #endif
        }

        m[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] = max;

        // Preparing for softmax by updating entry values and summing them up
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          DataType& current = P_S[q_index*K_SIZE + k_index];
          if (current <= masking_value)
            current = DataType(0);
          else
            current = TypeUtil<DataType>::Exp((current - max));
          sum += static_cast<L_DataType>(current);
        }

        l[b_index*H_SIZE*Q_SIZE + h_index*Q_SIZE + q_index] = sum;

        // Doing Normalisation to turn S into P containing a proper distribution,
        // and computing O = P @ V and dV = P^T @ dO
        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          const auto p = P_S[q_index*K_SIZE + k_index] = static_cast<DataType>(static_cast<L_DataType>(P_S[q_index*K_SIZE + k_index]) / sum);

          for (int d_index = 0; d_index < V_D_SIZE; ++d_index) {
            O[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index] += p * V[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index];

            ans_dV[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + k_index] += p * dO[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index];

          }
        }

        // Computing dS first, then
        // dQ = dS @ K and dK = dS^T @ Q
        DataType D(0);
        for (int d_index = 0; d_index < V_D_SIZE; ++d_index) {
          D += dO[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index] * O[b_index*H_SIZE*Q_SIZE*V_D_SIZE + h_index*Q_SIZE*V_D_SIZE + d_index * Q_SIZE + q_index];
        }

        for (int k_index = 0; k_index < K_SIZE; ++k_index) {
          d_P_S[q_index*K_SIZE + k_index] = P_S[q_index*K_SIZE + k_index] * (d_P_S[q_index*K_SIZE + k_index] - D) * dot_scaler;

          for (int d_index = 0; d_index < D_SIZE; ++d_index) {

            ans_dQ[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index] += d_P_S[q_index*K_SIZE + k_index] * K[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index];

            ans_dK[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + k_index] += d_P_S[q_index*K_SIZE + k_index] * Q[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + q_index];

          }
        }
      }
    }
  }

  // Doing the GPU-version of backward
  thrust::device_vector<DataType> dev_Q = Q;
  thrust::device_vector<DataType> dev_dQ = dQ;
  thrust::device_vector<DataType> dev_K = K;
  thrust::device_vector<DataType> dev_dK = dK;
  thrust::device_vector<DataType> dev_V = V;
  thrust::device_vector<DataType> dev_dV = dV;
  thrust::device_vector<DataType> dev_O = O;
  thrust::device_vector<DataType> dev_dO = dO;
  thrust::device_vector<L_DataType> dev_l = l;
  thrust::device_vector<DataType> dev_m = m;

  thrust::device_vector<uint32_t> Br_occupancy(B_SIZE*H_SIZE*launcher.ComputeNumOfBrSections(Q_SIZE), 0);

  cudaStream_t stream_handle = nullptr;
  Stopwatch stopwatch(stream_handle);

  stopwatch.start();
  cudaError_t launch_result = launcher.Backward(
        stream_handle, SharedMemoryDescriptor(48<<10, 48<<10),
        B_SIZE*H_SIZE,
        Q_SIZE, K_SIZE,
        D_SIZE, V_D_SIZE,
        dev_Q.data().get(),
        dev_K.data().get(),
        dev_V.data().get(),
        dev_O.data().get(),
        dev_l.data().get(),
        dev_m.data().get(),
        dev_dO.data().get(),
        dev_dQ.data().get(),
        dev_dK.data().get(),
        dev_dV.data().get(),
        Br_occupancy.data().get(),
        reference_seq_shape,
        Q_seq_order_map, K_seq_order_map,
        attention_policy);
  stopwatch.stop();

  std::stringstream message;
  if (cudaError_t::cudaSuccess != launch_result) {
    message.str("");
    message << "Failed to launch the Forward kernel: "
            << cudaGetErrorString(launch_result)
            << "(" << launch_result << ")";
    throw std::runtime_error(message.str());
  }

  // Calling stopwatch.seconds() will enforce the synchronisation between the host and the device,
  // so no need to invoke cudaDeviceSynchronize()
  //cudaDeviceSynchronize();
  float elapsed_seconds = stopwatch.seconds();
  printf("kernel running time: %f secs\n", elapsed_seconds);

  dQ = dev_dQ;
  dK = dev_dK;
  dV = dev_dV;


  // Comparing dQ, dK, and dV against ans_dQ, ans_dK, and ans_dV
  //
  // To account for precision errors which might get enlarged through accumulation,
  // normalisation by the number of entries from which the result is computed is conducted
  float error_count_dQ = 0.0, error_count_dK = 0.0, error_count_dV = 0.0;
  float max_error_dQ = 0.0, max_error_dK = 0.0, max_error_dV = 0.0;
  for (int b_index = 0; b_index < B_SIZE; ++b_index) {
    for (int h_index = 0; h_index < H_SIZE; ++h_index) {
      for (int seq_index = 0; seq_index < max(Q_SIZE, K_SIZE); ++seq_index) {

        for (int d_index = 0; d_index < max(D_SIZE, V_D_SIZE); ++d_index) {

          if (seq_index < Q_SIZE && d_index < D_SIZE) {
            const float error_dQ = std::abs(static_cast<float>(dQ[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + seq_index] - ans_dQ[b_index*H_SIZE*Q_SIZE*D_SIZE + h_index*Q_SIZE*D_SIZE + d_index * Q_SIZE + seq_index])) / K_SIZE;
            error_count_dQ += static_cast<float>(error_dQ > Precision<DataType>::value);

            max_error_dQ = max(error_dQ, max_error_dQ);
          }

          if (seq_index < K_SIZE) {
            if (d_index < D_SIZE) {
              const float error_dK = std::abs(static_cast<float>(dK[b_index*H_SIZE*K_SIZE*D_SIZE  + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + seq_index] - ans_dK[b_index*H_SIZE*K_SIZE*D_SIZE + h_index*K_SIZE*D_SIZE + d_index * K_SIZE + seq_index])) / Q_SIZE;
              error_count_dK += static_cast<float>(error_dK > Precision<DataType>::value);

              max_error_dK = max(error_dK, max_error_dK);
            }

            if (d_index < V_D_SIZE) {
              const float error_dV = std::abs(static_cast<float>(dV[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + seq_index] - ans_dV[b_index*H_SIZE*K_SIZE*V_D_SIZE + h_index*K_SIZE*V_D_SIZE + d_index * K_SIZE + seq_index])) / Q_SIZE;
              error_count_dV += static_cast<float>(error_dV > Precision<DataType>::value);

              max_error_dV = max(error_dV, max_error_dV);
            }

          }

        }
      }

    }
  }

  printf("error rate: dQ = %f, dK = %f, dV = %f\n",
          error_count_dQ/dQ.size(),
          error_count_dK/dK.size(),
          error_count_dV/dV.size());
  printf("max error: dQ = %f, dK = %f, dV = %f\n",
          max_error_dQ,
          max_error_dK,
          max_error_dV);
}

} // anonymous namespace


int main(int argc, char** argv)
{
  using namespace cute;

  //using DataType = half;
  //using DataType = float;
  //using DataType = double;

  const auto random_seed = time(nullptr);
  printf("random_seed = %lu\n", random_seed);
  srand(random_seed);

  const int device_id = 0;
  cudaDeviceProp device_prop;
  CUTE_CHECK_ERROR(cudaSetDevice(device_id));
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device_id));

  printf("Testing on GPU device %d (%s), CC = %d.%d\n",
          device_id, device_prop.name, device_prop.major, device_prop.minor);

  // TestForward()/TestBackward() checks the attention implementation
  // under the following configuration:
  //
  // data type = INTERNAL_TEST
  // dimensionality = 1d
  // attention_policy = full

  TestForward<INTERNAL_TEST>();
  TestBackward<INTERNAL_TEST>();

  return 0;
}


