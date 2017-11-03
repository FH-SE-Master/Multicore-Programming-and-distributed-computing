//       $Id: pfc_cuda_dim3.h 1306 2017-11-03 09:43:10Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/se/sw/mpv3/trunk/Lecture/Source/ACC/common/pfc_cuda_dim3.h $
// $Revision: 1306 $
//     $Date: 2017-11-03 10:43:10 +0100 (Fr., 03 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_CUDA_DIM3_H
#define      PFC_CUDA_DIM3_H

#include "./pfc_cuda_macros.h"
#include "./pfc_traits.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace pfc { namespace cuda {   // pfc::cuda

// -------------------------------------------------------------------------------------------------

struct dim3 final {
   using value_t = int;

   template <typename X = value_t, typename Y = value_t, typename Z = value_t> CATTR_HOST_DEV constexpr dim3 (X const x = 1, Y const y = 1, Z const z = 1) noexcept
      : x {static_cast <value_t> (x)}
      , y {static_cast <value_t> (y)}
      , z {static_cast <value_t> (z)} {

      static_assert (pfc::is_integral_v <X>, "X must be a pfc-integral type");
      static_assert (pfc::is_integral_v <Y>, "Y must be a pfc-integral type");
      static_assert (pfc::is_integral_v <Z>, "Z must be a pfc-integral type");
   }

   CATTR_HOST_DEV constexpr dim3 (uint3 const & d) noexcept
      : x {static_cast <value_t> (d.x)}
      , y {static_cast <value_t> (d.y)}
      , z {static_cast <value_t> (d.z)} {
   }

   CATTR_HOST_DEV constexpr dim3 (::dim3 const & d) noexcept
      : x {static_cast <value_t> (d.x)}
      , y {static_cast <value_t> (d.y)}
      , z {static_cast <value_t> (d.z)} {
   }

   CATTR_HOST_DEV constexpr operator uint3 () const noexcept {
      return {
         static_cast <decltype (uint3::x)> (x),
         static_cast <decltype (uint3::y)> (y),
         static_cast <decltype (uint3::z)> (z)
      };
   }

   CATTR_HOST_DEV operator ::dim3 () const noexcept {
      return {
         static_cast <decltype (::dim3::x)> (x),
         static_cast <decltype (::dim3::y)> (y),
         static_cast <decltype (::dim3::z)> (z)
      };
   }

   value_t x {1};
   value_t y {1};
   value_t z {1};
};

}   // namespace cuda

namespace cuda_literals {

// -------------------------------------------------------------------------------------------------

CATTR_HOST_DEV constexpr inline pfc::cuda::dim3 operator "" _dim3 (unsigned long long const literal) {
   return literal;
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda_literals

#endif   // PFC_CUDA_DIM3_H
