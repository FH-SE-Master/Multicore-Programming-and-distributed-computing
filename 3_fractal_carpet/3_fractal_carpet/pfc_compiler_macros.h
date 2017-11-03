//       $Id: pfc_compiler_macros.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_compiler_macros.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_COMPILER_MACROS_H
#define      PFC_COMPILER_MACROS_H

// -------------------------------------------------------------------------------------------------

#undef COMP_CL
#undef COMP_CLANG
#undef COMP_GCC
#undef COMP_INTEL
#undef COMP_MICROSOFT
#undef COMP_NVCC

// -------------------------------------------------------------------------------------------------

#if defined __clang__
   #define COMP_CLANG
#endif

#if defined __CUDACC__
   #define COMP_NVCC
#endif

#if defined __GNUC__
   #define COMP_GCC
#endif

#if defined __INTEL_COMPILER
   #define COMP_INTEL
#endif

#if defined _MSC_VER
   #define COMP_CL
   #define COMP_MICROSOFT
#endif

// -------------------------------------------------------------------------------------------------

#endif   // PFC_COMPILER_MACROS_H
