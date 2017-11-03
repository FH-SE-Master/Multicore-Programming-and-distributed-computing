//       $Id: pfc_rgb_from_wavelength.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_rgb_from_wavelength.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_RGB_FROM_WAVELENGTH_H
#define      PFC_RGB_FROM_WAVELENGTH_H

#include "./pfc_units.h"

using namespace pfc::literals;

namespace pfc {

// -------------------------------------------------------------------------------------------------

namespace {

auto const nm_380 {380_nm};   // UV - violett
auto const nm_780 {780_nm};   // rot - IR

}

// -------------------------------------------------------------------------------------------------

template <typename color_t> void rgb_from_wavelength (color_t & c, pfc::nanometer_t const wl, bool const gray = false, double const alpha = 0.8) {
   using blue_t  = decltype (color_t::blue);
   using green_t = decltype (color_t::green);
   using red_t   = decltype (color_t::red);

   using col_comp_t = blue_t;

   static_assert (pfc::is_integral_v <col_comp_t>);

   static_assert (std::is_same_v <col_comp_t, green_t>);
   static_assert (std::is_same_v <col_comp_t, red_t>);

   auto const a {std::clamp (alpha, 0.0, 1.0)};
   auto const w {pfc::unit_value_cast <pfc::picometer_t, int> (std::clamp (wl, pfc::nm_380, pfc::nm_780))};

   int b {0};
   int g {0};
   int r {0};

   if (w <  440000) { r = (440000 - w) / 60; b = 1000; } else
   if (w <  490000) { g = (w - 440000) / 50; b = 1000; } else
   if (w <  510000) { g = 1000; b = (510000 - w) / 20; } else
   if (w <  580000) { r = (w - 510000) / 70; g = 1000; } else
   if (w <  645000) { r = 1000; g = (645000 - w) / 65; } else
   if (w <= 780000) { r = 1000; }

   double f {0};

   if (w <  420000) { f = -0.006350 + 0.00000001750 * w; } else
   if (w <  701000) { f =  0.001000; } else
   if (w <= 780000) { f =  0.007125 - 0.00000000875 * w; }

   constexpr auto       max {pfc::limits_max_v <col_comp_t>};
   constexpr col_comp_t min {};

   c.blue  = pfc::clamp_indirect (std::pow (b * f, a), min, max);
   c.green = pfc::clamp_indirect (std::pow (g * f, a), min, max);
   c.red   = pfc::clamp_indirect (std::pow (r * f, a), min, max);

   if (gray) {
      c.red = c.green = c.blue = static_cast <col_comp_t> (0.299 * c.red + 0.587 * c.green + 0.114 * c.blue);
   }
}

template <typename color_t> void rgb_from_wavelength (color_t & c, double const x, bool const gray = false, double const alpha = 0.8) {
   rgb_from_wavelength (c, pfc::clamp_indirect (x, pfc::nm_380, pfc::nm_780), gray, alpha);
}

// -------------------------------------------------------------------------------------------------

}   // namespace pfc

#endif   // PFC_RGB_FROM_WAVELENGTH_H
