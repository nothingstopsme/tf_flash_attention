#ifndef __TF_TYPE_DECLARATION_H__
#define __TF_TYPE_DECLARATION_H__

#include "tensorflow/core/framework/types.h"
#include <cuda_fp16.h>

namespace tensorflow {
template <>
struct DataTypeToEnum<half> {
  static DataType v() { return value; }
  static DataType ref() { return MakeRefType(value); }
  static constexpr DataType value = DT_HALF;
};
template <>
struct IsValidDataType<half> {
  static constexpr bool value = true;
};

} // namespace tensorflow

#endif //__TF_TYPE_DECLARATION_H__
