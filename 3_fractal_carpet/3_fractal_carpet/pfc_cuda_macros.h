//       $Id: pfc_cuda_macros.h 1231 2017-10-12 18:46:26Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/se/sw/mpv3/trunk/Lecture/Source/ACC/common/pfc_cuda_macros.h $
// $Revision: 1231 $
//     $Date: 2017-10-12 20:46:26 +0200 (Do., 12 Okt 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_CUDA_MACROS_H
#define      PFC_CUDA_MACROS_H

#include "./pfc_compiler_macros.h"

// -------------------------------------------------------------------------------------------------

#undef CARCH
#undef CARCH_100
#undef CARCH_110
#undef CARCH_120
#undef CARCH_130
#undef CARCH_200
#undef CARCH_210
#undef CARCH_300
#undef CARCH_320
#undef CARCH_350
#undef CARCH_370
#undef CARCH_500
#undef CARCH_520
#undef CARCH_520
#undef CARCH_530
#undef CARCH_600
#undef CARCH_610
#undef CARCH_620
#undef CARCH_700
#undef CARCH_710

#undef CATTR_CONST
#undef CATTR_DEV_CONST
#undef CATTR_DEVICE
#undef CATTR_FINLINE
#undef CATTR_GLOBAL
#undef CATTR_HOST
#undef CATTR_HOST_DEV
#undef CATTR_HOST_DEV_INL
#undef CATTR_KERNEL
#undef CATTR_LBOUNDS
#undef CATTR_RESTRICT
#undef CATTR_SHARED

// -------------------------------------------------------------------------------------------------

#if defined __CARCH__
   #define CARCH

   #if   __CARCH__ >= 710
      #define CARCH_710

   #elif __CARCH__ >= 700
      #define CARCH_700

   #elif __CARCH__ >= 620
      #define CARCH_620

   #elif __CARCH__ >= 610
      #define CARCH_610

   #elif __CARCH__ >= 600
      #define CARCH_600

   #elif __CARCH__ >= 530
      #define CARCH_530

   #elif __CARCH__ >= 370
      #define CARCH_370

   #elif __CARCH__ >= 530
      #define CARCH_530

   #elif __CARCH__ >= 520
      #define CARCH_520

   #elif __CARCH__ >= 500
      #define CARCH_500

   #elif __CARCH__ >= 370
      #define CARCH_370

   #elif __CARCH__ >= 350
      #define CARCH_350

   #elif __CARCH__ >= 320
      #define CARCH_320

   #elif __CARCH__ >= 300
      #define CARCH_300

   #elif __CARCH__ >= 210
      #define CARCH_210

   #elif __CARCH__ >= 200
      #define CARCH_200

   #elif __CARCH__ >= 130
      #define CARCH_130

   #elif __CARCH__ >= 120
      #define CARCH_120

   #elif __CARCH__ >= 110
      #define CARCH_110

   #elif __CARCH__ >= 100
      #define CARCH_100

   #endif
#endif

// -------------------------------------------------------------------------------------------------

#if defined COMP_NVCC && defined __constant__
   #define CATTR_CONST __constant__
#else
   #define CATTR_CONST
#endif

#if defined COMP_NVCC && defined __device__
   #define CATTR_DEVICE __device__
#else
   #define CATTR_DEVICE
#endif

#if defined COMP_NVCC && defined __forceinline__
   #define CATTR_FINLINE __forceinline__
#else
   #define CATTR_FINLINE inline
#endif

#if defined COMP_NVCC && defined __global__
   #define CATTR_GLOBAL __global__
#else
   #define CATTR_GLOBAL
#endif

#if defined COMP_NVCC && defined __host__
   #define CATTR_HOST __host__
#else
   #define CATTR_HOST
#endif

#if defined COMP_NVCC && defined __launch_bounds__
   #define CATTR_LBOUNDS(tpb, bpm) __launch_bounds__ (tpb, bpm)   /* tpb ... max. threads per block         */
#else                                                             /* bpm ... min. blocks per multiprocessor */
   #define CATTR_LBOUNDS(tpb, bpm)
#endif

#if defined COMP_NVCC && defined __restrict__
   #define CATTR_RESTRICT __restrict__
#else
   #define CATTR_RESTRICT __restrict
#endif

#if defined COMP_NVCC && defined __shared__
   #define CATTR_SHARED __shared__
#else
   #define CATTR_SHARED
#endif

// -------------------------------------------------------------------------------------------------

#define CATTR_DEV_CONST    CATTR_DEVICE CATTR_CONST
#define CATTR_HOST_DEV     CATTR_HOST CATTR_DEVICE
#define CATTR_HOST_DEV_INL CATTR_HOST CATTR_DEVICE CATTR_FINLINE
#define CATTR_KERNEL       CATTR_GLOBAL

// -------------------------------------------------------------------------------------------------

#endif   // PFC_CUDA_MACROS_H
