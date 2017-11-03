//       $Id: pfc_timer.h 35946 2017-10-31 22:51:39Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_timer.h $
// $Revision: 35946 $
//     $Date: 2017-10-31 23:51:39 +0100 (Di., 31 Okt 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_TIMER
#define      PFC_TIMER

#include "./pfc_base.h"

#if defined _WINDOWS_
   #undef  PFC_WINDOWS_H_INCLUDED
   #define PFC_WINDOWS_H_INCLUDED
#endif

#include <chrono>
#include <functional>

namespace pfc {

// -------------------------------------------------------------------------------------------------

#if defined PFC_WINDOWS_H_INCLUDED

struct tsc_clock final {
   using duration   = std::chrono::duration <decltype (__rdtsc ()), std::ratio <1, 1>>;
   using period     = duration::period;
   using rep        = duration::rep;
   using time_point = std::chrono::time_point <pfc::tsc_clock>;

   constexpr static bool const is_steady {false};

   static time_point now () {
      return time_point {duration {__rdtsc ()}};
   }
};

#endif   // PFC_WINDOWS_H_INCLUDED

using default_clock_t = std::chrono::steady_clock;

// -------------------------------------------------------------------------------------------------

template <typename duration_t> auto in_s (duration_t const & duration) {
   return pfc::prefix_cast <typename duration_t::period> (duration.count ());
}

template <typename duration_t> constexpr char const * time_unit () noexcept {
   using period_t = typename duration_t::period;

   if (std::is_same_v <period_t, std::nano>)
      return "ns";

   else if (std::is_same_v <period_t, pfc::hectonano>)
      return "hns";

   else if (std::is_same_v <period_t, std::micro>)
      return "us";

   else if (std::is_same_v <period_t, std::milli>)
      return "ms";

   else if (std::is_same_v <period_t, std::ratio <1, 1>>)
      return "s";

   else if (std::is_same_v <period_t, std::ratio <1, 60>>)
      return "min";

   else if (std::is_same_v <period_t, std::ratio <1, 60 * 60>>)
      return "h";

   else
      return "?";
}

template <typename duration_t> constexpr auto time_unit (duration_t const &) noexcept {
   return pfc::time_unit <duration_t> ();
}

// -------------------------------------------------------------------------------------------------

template <typename clock_t = pfc::default_clock_t, typename size_t, typename fun_t, typename ...args_t> auto timed_run (size_t const n, fun_t && fun, args_t && ...args) noexcept (std::is_nothrow_invocable_v <fun_t, args_t...>) {
   static_assert (clock_t::is_steady,          "clock_t must denote a steady clock");
   static_assert (std::is_integral_v <size_t>, "size_t must be an integral type");

   typename clock_t::duration elapsed {};

   if (0 < n) {
      auto const start {clock_t::now ()};

      for (int i {0}; i < n; ++i) {
         std::invoke (std::forward <fun_t> (fun), std::forward <args_t> (args)...);
      }

      elapsed = (clock_t::now () - start) / n;
   }

   return elapsed;
}

template <typename clock_t = default_clock_t, typename fun_t, typename ...args_t> auto timed_run (fun_t && fun, args_t && ...args) noexcept (std::is_nothrow_invocable_v <fun_t, args_t...>) {
   return pfc::timed_run (1, std::forward <fun_t> (fun), std::forward <args_t> (args)...);
}

// -------------------------------------------------------------------------------------------------

template <typename rep_t, typename period_t> std::ostream & operator << (std::ostream & lhs, std::chrono::duration <rep_t, period_t> const & rhs) {
   return lhs << rhs.count () << pfc::time_unit (rhs);
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_TIMER
