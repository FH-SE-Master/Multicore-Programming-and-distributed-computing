//       $Id: pfc_cuda_exception.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_cuda_exception.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_CUDA_EXCEPTION_H
#define      PFC_CUDA_EXCEPTION_H

#include "./pfc_cuda_macros.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>

using namespace std::literals;

// -------------------------------------------------------------------------------------------------

#undef  PFC_CUDA_CHECK
#define PFC_CUDA_CHECK(call) \
   pfc::cuda::check (call, __FILE__, __LINE__)

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace cuda {   // pfc::cuda

class exception final : public std::runtime_error {
   using inherited = std::runtime_error;

   static auto make_message (cudaError_t const error, std::string const & file, int const line) {
      auto message {"CUDA error #"s};

      message += std::to_string (error);
      message += " '";
      message += cudaGetErrorString (error);
      message += "' occurred";

      if (!file.empty () && (line > 0)) {
         message += " in file '";
         message += file;
         message += "' on line ";
         message += std::to_string (line);
      }

      return std::move (message);
   }

   public:
      explicit exception (cudaError_t const error, std::string const & file, int const line)
         : inherited {make_message (error, file, line)} {
      }
};

// -------------------------------------------------------------------------------------------------

inline void check (cudaError_t const error, std::string const & file, int const line) {
   if (error != cudaSuccess) {
      throw pfc::cuda::exception (error, file, line);
   }
}

inline void check (cudaError_t const error) {
   pfc::cuda::check (error, "", 0);
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda

#endif   // PFC_CUDA_EXCEPTION_H
