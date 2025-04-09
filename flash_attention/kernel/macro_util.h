#ifndef __MACRO_UNTIL_H__
#define __MACRO_UNTIL_H__

#ifndef HALF_TYPE_ITERATOR
#error "HALF_TYPE_ITERATOR is not defined!"
#endif

#define ITERATE_SEQUENCE_DIMS(func, ...) \
  func(__VA_ARGS__, 1) \
  func(__VA_ARGS__, 2)

#define ITERATE_TYPES(func, ...) \
  func(__VA_ARGS__, HALF_TYPE_ITERATOR) \
  func(__VA_ARGS__, float) \
  func(__VA_ARGS__, double)

#define ITERATE_TYPES_NO_HALF(func, ...) \
  func(__VA_ARGS__, float) \
  func(__VA_ARGS__, double)

#define ITERATE_TYPES_SEQUENCE_DIMS(...) \
  ITERATE_TYPES(ITERATE_SEQUENCE_DIMS, __VA_ARGS__)

#define ITERATE_TYPES_NO_HALF_SEQUENCE_DIMS(...) \
  ITERATE_TYPES_NO_HALF(ITERATE_SEQUENCE_DIMS, __VA_ARGS__)

#endif //__MACRO_UNTIL_H__
