#ifndef __TYPE_MAPPING_H__
#define __TYPE_MAPPING_H__

#include <cuda_fp16.h>

template <size_t FP_Size> struct FP_TypeMapping;
template <> struct FP_TypeMapping<2> { using FP_Type = half; };
template <> struct FP_TypeMapping<4> { using FP_Type = float; };
template <> struct FP_TypeMapping<8> { using FP_Type = double; };

template <size_t UI_Size> struct UI_TypeMapping {};
template <> struct UI_TypeMapping<2> { using UI_Type = uint16_t; };
template <> struct UI_TypeMapping<4> { using UI_Type = uint32_t; };
template <> struct UI_TypeMapping<8> { using UI_Type = uint64_t; };
//template <> struct UI_TypeMapping<16> { using UI_Type = cute::uint128_t; };



#endif //__TYPE_MAPPING_H__
