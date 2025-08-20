#ifndef __CUTE_EXT_BOUNDARY_CHECK_PRED_H__
#define __CUTE_EXT_BOUNDARY_CHECK_PRED_H__

#include <cute/int_tuple.hpp>

namespace cute_ext {

template <size_t I = 0, typename A, typename B> //, typename Comparator>
__device__
inline constexpr bool WithinBoundary(const A& a, const B& b) {


  if constexpr (cute::is_tuple<A>::value) {
    static_assert(cute::is_tuple<B>::value, "The \"tuple-single element\" comparison is not supported");

    static_assert(cute::tuple_size<A>::value == cute::tuple_size<B>::value,
                    "Tuples for comparison must have an equal length in this setting");
    if constexpr (cute::tuple_size<A>::value == I)
      return true;
    else
      return WithinBoundary(cute::get<I>(a), cute::get<I>(b)) && WithinBoundary<I+1>(a, b);
      //return CompareDims(get<I>(a), get<I>(b), comparator) && CompareDims<I+1>(a, b, comparator);
  }
  else {
    static_assert(cute::is_tuple<B>::value && decltype(cute::depth(b))::value == 1 && cute::tuple_size<B>::value == 2, "b should be a boundary tuple");

    return a >= cute::get<0>(b) && a <= cute::get<1>(b);
    //return comparator(a, b);
  }

  CUTE_GCC_UNREACHABLE;
}


template <typename Mapping, typename BBox, size_t... Dimensions>
class BoundaryCheckPred {
 public:

  __device__
  inline BoundaryCheckPred(Mapping&& mapping, BBox&& bbox)
  : _mapping(static_cast<Mapping&&>(mapping)),
    _bbox(static_cast<BBox&&>(bbox)) {

    static_assert(decltype(cute::depth(_bbox))::value == 2
					&& decltype(cute::rank(_bbox))::value == sizeof...(Dimensions)
					&& decltype(cute::rank(_bbox))::value * 2 == cute::tuple_size<decltype(cute::flatten(_bbox))>::value,
                  "bbox should have the form of ((lower_1, upper_1), ..., (lower_N, upper_N)) with N equal to the size of the template parameter pack \"Dimensions\"");
  }

  __device__
  inline BoundaryCheckPred(Mapping&& mapping, const BBox& bbox)
  : BoundaryCheckPred(static_cast<Mapping&&>(mapping), BBox(bbox)) {
  }



  template <typename... Coords>
  __device__
  inline constexpr auto operator()(const Coords&... coords) const {

    auto lookup = _mapping(coords...);
    using LookupType = decltype(lookup);

    if constexpr (HasUnderscore<Coords...>())
      return BoundaryCheckPred<LookupType, BBox, Dimensions...>(static_cast<LookupType&&>(lookup), _bbox);
    else {
      const auto targets = cute::select<Dimensions...>(lookup);
      return WithinBoundary(targets, _bbox);
    }
  }

  template <typename... Coords>
  __device__
  inline constexpr void Print(const Coords&... coords) const {

    auto lookup = _mapping(coords...);
    cute::print(lookup); cute::print("/"); cute::print(_bbox);
    cute::print(" = %d\n", this->operator()(coords...));
  }


  template <int32_t B, int32_t E>
  __device__
  inline constexpr auto group_modes() const {

    auto grouped = cute::group_modes<B,E>(_mapping);
    using GroupedType = decltype(grouped);

    return BoundaryCheckPred<GroupedType, BBox, Dimensions...>(static_cast<GroupedType&&>(grouped), _bbox);

  }


 private:
  template <typename Coord0, typename... OtherCoords>
  __device__
  static inline constexpr bool HasUnderscore() {
    if constexpr (sizeof...(OtherCoords) == 0) {
      return cute::has_underscore<Coord0>::value;
    }
    else
      return cute::has_underscore<Coord0>::value ||  HasUnderscore<OtherCoords...>();
  }



  const Mapping _mapping;
  const BBox _bbox;
};

template <size_t... Dimensions, typename Mapping, typename BBoxMin, typename BBoxMax>
__device__
inline constexpr
auto GetBoundaryCheckPred(Mapping&& mapping, BBoxMin&& bbox_min, BBoxMax&& bbox_max) {
  auto bbox = cute::zip(bbox_min, bbox_max);
  using BBox = decltype(bbox);
  return BoundaryCheckPred<Mapping, BBox, Dimensions...>(
          static_cast<Mapping&&>(mapping),
          static_cast<BBox&&>(bbox));
}

template <size_t... Dimensions, typename Mapping, typename Shape>
__device__
inline constexpr
auto GetBoundaryCheckPred(Mapping&& mapping, const Shape& shape) {
  const auto picked_dims = cute::select<Dimensions...>(shape);
  constexpr auto dims_rank = decltype(rank(picked_dims))::value;

  auto bbox = cute::zip(cute::tuple_repeat<dims_rank>(cute::Int<0>{}),
                  cute::transform(picked_dims, cute::tuple_repeat<dims_rank>(cute::Int<1>{}), cute::minus{}));


  using BBox = decltype(bbox);
  return BoundaryCheckPred<Mapping, BBox, Dimensions...>(
          static_cast<Mapping&&>(mapping),
          static_cast<BBox&&>(bbox));
}

#if 0
template <int32_t B, int32_t E, typename Mapping, typename BBox, int... Dimensions>
__device__
static inline constexpr auto group_modes(const BoundaryCheckPred<Mapping, BBox, Dimensions...>& pred) {
  return pred.template group_modes<B, E>();
}
#endif

} // namespace cute_ext

#endif // __BOUNDARY_CHECK_PRED_H__
