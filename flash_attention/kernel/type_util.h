#ifndef __TYPE_UTIL_H__
#define __TYPE_UTIL_H__

#include "type_mapping.h"

template <typename T>
struct TypeUtil {
 private:
  using UI_Type = typename UI_TypeMapping<sizeof(T)>::UI_Type;

  template <uint8_t Value, UI_Type Current = 0, size_t I = 0>
  CUTE_HOST_DEVICE
  static constexpr UI_Type RepeatByte() {
    if constexpr (I < sizeof(UI_Type)) {
      return RepeatByte<Value, Current << 8 | Value, I+1>();
    }
    else
      return Current;
  }

  CUTE_HOST_DEVICE
  static constexpr T MapFromBitfield(const UI_Type& raw) {
    if constexpr (std::is_same<T, half>::value) {
      return half(__half_raw{raw});
    }
    else if constexpr (std::is_same<T, half2>::value) {
      return half2(__half_raw{
                    static_cast<uint16_t>(raw & ((decltype(raw)(1) << 16) - 1))
                  },
                  __half_raw{
                    static_cast<uint16_t>(raw >> 16)
                  });
    }
    else {
      // Note that reinterpret_cast() is not a constexpr op;
      // so for data types with higher precision, MapFromBitfield() is never a constexpr call
      return *reinterpret_cast<const T*>(&raw);
    }
  }

 public:
  CUTE_HOST_DEVICE
  static constexpr T GetNegInfApprox() {
    return MapFromBitfield(RepeatByte<0xFA>());
  }

  CUTE_HOST_DEVICE
  static constexpr T Exp(const T& x) {
    if constexpr (std::is_same<T, half>::value) {
      #if (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA)
      return hexp(x);
      #else
      return static_cast<half>(exp(static_cast<float>(x)));
      #endif
    }
    else if constexpr (std::is_same<T, half2>::value) {
      #if (__CUDA_ARCH__ >= 530) || defined(_NVHPC_CUDA)
      return h2exp(x);
      #else
      return half2(
              static_cast<half>(exp(static_cast<float>(x.x))),
              static_cast<half>(exp(static_cast<float>(x.y)))
              );
      #endif
    }
    else
      return exp(x);
  }

  CUTE_HOST_DEVICE
  static constexpr T RSqrt(const T& x) {
    if constexpr (std::is_same<T, half>::value) {
      #if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
      return hrsqrt(x);
      #else
      return static_cast<half>(rsqrt(static_cast<float>(x)));
      #endif
    }
    else if constexpr (std::is_same<T, half2>::value) {
      #if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
      return h2rsqrt(x);
      #else
      return half2(
              static_cast<half>(rsqrt(static_cast<float>(x.x))),
              static_cast<half>(rsqrt(static_cast<float>(x.y)))
              );
      #endif

    }
    else
      return rsqrt(x);
  }

  CUTE_HOST_DEVICE
  static constexpr T Sqrt(const T& x) {
    if constexpr (std::is_same<T, half>::value) {
      #if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
      return hsqrt(x);
      #else
      return static_cast<half>(sqrt(static_cast<float>(x)));
      #endif
    }
    else if constexpr (std::is_same<T, half2>::value) {
      #if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
      return h2sqrt(x);
      #else
      return half2(
              static_cast<half>(sqrt(static_cast<float>(x.x))),
              static_cast<half>(sqrt(static_cast<float>(x.y)))
              );
      #endif

    }
    else
      return sqrt(x);
  }


  CUTE_HOST_DEVICE
  static constexpr T Abs(const T& x) {
    if constexpr (std::is_same<T, half>::value) {
      return __habs(x);
    }
    else if constexpr (std::is_same<T, half2>::value) {
      return __habs2(x);
    }
    else
      return abs(x);
  }

  CUTE_HOST_DEVICE
  static constexpr T Max(const T& x, const T& y) {
    if constexpr (std::is_same<T, half>::value) {
      return __hmax(x, y);
    }
    else if constexpr (std::is_same<T, half2>::value) {
      return __hmax2(x, y);
    }
    else
      return max(x, y);
  }

  CUTE_HOST_DEVICE
  static constexpr T Min(const T& x, const T& y) {
    if constexpr (std::is_same<T, half>::value) {
      return __hmin(x, y);
    }
    else if constexpr (std::is_same<T, half2>::value) {
      return __hmin2(x, y);
    }
    else
      return min(x, y);
  }

};


#endif // __TYPE_UTIL_H__
