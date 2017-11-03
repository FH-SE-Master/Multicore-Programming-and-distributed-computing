//       $Id: pfc_base.h 35950 2017-11-01 14:04:10Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_base.h $
// $Revision: 35950 $
//     $Date: 2017-11-01 15:04:10 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_BASE_H
#define      PFC_BASE_H

#include "./pfc_cuda_macros.h"
#include "./pfc_traits.h"

#include <algorithm>

namespace pfc {

// -------------------------------------------------------------------------------------------------

template <typename value_t> constexpr auto clamp (value_t const & value, value_t const & left, value_t const & right) noexcept {
   return std::max (left, std::min (right, value));
}

template <typename value_t> constexpr auto clamp_indirect (double const f, value_t const & left, value_t const & right) {
   return static_cast <value_t> (left + (right - left) * pfc::clamp (f, 0.0, 1.0));   // !pwk: use std::clamp
}

template <typename A, typename B> constexpr auto ceil_div (A const a, B const b) noexcept {
   static_assert (pfc::is_integral_v <A>, "A must be a pfc-integral type");
   static_assert (pfc::is_integral_v <B>, "B must be a pfc-integral type");

   if /*constexpr*/ (pfc::is_integral_unsigned_v <A>) {
      return             (b > 0) ? (a + b - 1) / b : 0;
   } else {
      return (a >= 0) && (b > 0) ? (a + b - 1) / b : 0;
   }
}

template <typename T> constexpr int size (T const & t) noexcept {
   return static_cast <int> (std::size (t));
}

// -------------------------------------------------------------------------------------------------

template <typename ratio_t, typename value_t> constexpr auto prefix_cast (value_t const & value) noexcept {
   return static_cast <double> (value) * ratio_t::num / ratio_t::den;
}

template <typename enum_t> constexpr auto underlying_cast (enum_t const & value) noexcept {
   return static_cast <std::underlying_type_t <enum_t>> (value);
}

// -------------------------------------------------------------------------------------------------

using hectonano = std::ratio <1, std::nano::den / 100>;   // 100 nanos

template <typename ratio_t> constexpr char const * unit_prefix () noexcept {
   static_assert (pfc::is_ratio_v <ratio_t>, "ratio_t must be a std::ratio");

   if (std::is_same_v <ratio_t, std::atto>)
      return "a";

   else if (std::is_same_v <ratio_t, std::femto>)
      return "f";

   else if (std::is_same_v <ratio_t, std::pico>)
      return "p";

   else if (std::is_same_v <ratio_t, std::nano>)
      return "n";

   else if (std::is_same_v <ratio_t, pfc::hectonano>)
      return "hn";

   else if (std::is_same_v <ratio_t, std::micro>)
      return "u";

   else if (std::is_same_v <ratio_t, std::milli>)
      return "m";

   else if (std::is_same_v <ratio_t, std::centi>)
      return "c";

   else if (std::is_same_v <ratio_t, std::deci>)
      return "d";

   else if (std::is_same_v <ratio_t, std::deca>)
      return "da";

   else if (std::is_same_v <ratio_t, std::hecto>)
      return "h";

   else if (std::is_same_v <ratio_t, std::kilo>)
      return "k";

   else if (std::is_same_v <ratio_t, std::mega>)
      return "M";

   else if (std::is_same_v <ratio_t, std::giga>)
      return "G";

   else if (std::is_same_v <ratio_t, std::tera>)
      return "T";

   else if (std::is_same_v <ratio_t, std::peta>)
      return "P";

   else if (std::is_same_v <ratio_t, std::exa>)
      return "E";

   else
      return "?";
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_BASE_H
