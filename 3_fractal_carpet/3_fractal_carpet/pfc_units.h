//       $Id: pfc_units.h 35950 2017-11-01 14:04:10Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_units.h $
// $Revision: 35950 $
//     $Date: 2017-11-01 15:04:10 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_UNITS_H
#define      PFC_UNITS_H

#include "./pfc_base.h"

#include <boost/units/io.hpp>
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/prefixes.hpp>

namespace pfc {

// -------------------------------------------------------------------------------------------------

namespace {

auto const nano {boost::units::si::nano};
auto const pico {boost::units::si::pico};

auto const meter     {1.0 * boost::units::si::meter};
auto const nanometer {1.0 * pfc::nano * pfc::meter};
auto const picometer {1.0 * pfc::pico * pfc::meter};

}   // namespace

// -------------------------------------------------------------------------------------------------

using meter_t     = decltype (0.0 * boost::units::si::meter);
using nanometer_t = decltype (0.0 * boost::units::si::nano * boost::units::si::meter);
using picometer_t = decltype (0.0 * boost::units::si::pico * boost::units::si::meter);

// -------------------------------------------------------------------------------------------------

template <typename unit_to_t, typename unit_to_value_t, typename unit_from_t> auto unit_value_cast (unit_from_t const & unit) {
   return static_cast <unit_to_value_t> (static_cast <unit_to_t> (unit).value ());
}

// -------------------------------------------------------------------------------------------------

namespace literals {

inline auto operator "" _nm (unsigned long long const value) {
   return static_cast <double> (value) * pfc::nanometer;
}

inline auto operator "" _nm (long double const value) {
   return static_cast <double> (value) * pfc::nanometer;
}

inline auto operator "" _pm (unsigned long long const value) {
   return static_cast <double> (value) * pfc::picometer;
}

inline auto operator "" _pm (long double const value) {
   return static_cast <double> (value) * pfc::picometer;
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::literals

#endif   // PFC_UNITS_H
