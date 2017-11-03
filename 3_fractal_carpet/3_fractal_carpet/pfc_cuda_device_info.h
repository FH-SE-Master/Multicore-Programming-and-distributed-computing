//       $Id: pfc_cuda_device_info.h 35947 2017-11-01 09:51:53Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_cuda_device_info.h $
// $Revision: 35947 $
//     $Date: 2017-11-01 10:51:53 +0100 (Mi., 01 Nov 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_CUDA_DEVICE_INFO_H
#define      PFC_CUDA_DEVICE_INFO_H

#include "./pfc_cuda_exception.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace pfc { namespace cuda {   // pfc::cuda

// -------------------------------------------------------------------------------------------------

struct device_info final {
   int          cc_major              {0};    //  0
   int          cc_minor              {0};    //  1
   int          cores_sm              {0};    //  2
   char const * uarch                 {""};   //  4
   char const * chip                  {""};   //  5
   int          ipc                   {0};    //  6
   int          max_act_cores_sm      {0};    //  7
   int          max_regs_thread       {0};    //  8
   int          max_regs_block        {0};    //  9
   int          max_smem_block        {0};    // 10
   int          max_threads_block     {0};    // 11
   int          max_act_blocks_sm     {0};    // 12
   int          max_threads_sm        {0};    // 13
   int          max_warps_sm          {0};    // 14
   int          alloc_gran_regs       {0};    // 15
   int          regs_sm               {0};    // 16 (32-bit registers)
   int          alloc_gran_smem       {0};    // 17
   int          smem_bank_width       {0};    // 18
   int          smem_sm               {0};    // 19 (in bytes)
   int          smem_banks            {0};    // 20
   char const * sm_version            {""};   // 21
   int          warp_size             {0};    // 22 (in threads)
   int          alloc_gran_warps      {0};    // 23
   int          schedulers_sm         {0};    // 24
   int          width_cl1             {0};    // 25
   int          width_cl2             {0};    // 26
   int          load_store_units_sm   {0};    // 27
   int          load_store_throughput {0};    // 28 (per cycle)
   int          texture_units_sm      {0};    // 29
   int          texture_throughput    {0};    // 30 (per cycle)
   int          fp32_units_sm         {0};    // 31
   int          fp32_throughput       {0};    // 32 (per cycle)
   int          sf_units_sm           {0};    // 33 (special function unit, e.g. sin, cosine, square root)
   int          sfu_throughput        {0};    // 34 (per cycle)
};

// -------------------------------------------------------------------------------------------------

inline auto const & get_device_props (int const device = 0) noexcept {
   static cudaDeviceProp props; PFC_CUDA_CHECK (cudaGetDeviceProperties (&props, device)); return props;
}

/**
 * see <http://en.wikipedia.org/wiki/CUDA>
 * see <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities>
 * see <https://devblogs.nvidia.com/parallelforall/inside-volta>
 */
inline auto const & get_device_info (int const cc_major, int const cc_minor) {
   static std::map <std::tuple <int, int>, pfc::cuda::device_info> const info {
//              0  1    2  4          5            6   7    8      9     10    11  12    13  14   15      16   17  18     19  20  21       22  23  24   25  26  27  28  29  30   31   32  33  34
      {{0, 0}, {0, 0,   0, "",        "",          0,  0,   0,     0,     0,    0,  0,    0,  0,   0,      0,   0,  0,     0,  0, "",       0,  0,  0,   0,  0,  0,  0,  0,  0,   0,   0,  0,  0}},
      {{1, 0}, {1, 0,   8, "Tesla",   "G80",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_10", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 1}, {1, 1,   8, "Tesla",   "G8x",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_11", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 2}, {1, 2,   8, "Tesla",   "G9x",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_12", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{1, 3}, {1, 3,   8, "Tesla",   "GT20x",     1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_13", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1}},
      {{2, 0}, {2, 0,  32, "Fermi",   "GF10x",     1, 16,  63, 32768, 49152, 1024,  8, 1536, 48,  64,  32768, 128,  4, 49152, 32, "sm_20", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8}},
      {{2, 1}, {2, 1,  48, "Fermi",   "GF10x",     2, 16,  63, 32768, 49152, 1024,  8, 1536, 48,  64,  32768, 128,  4, 49152, 32, "sm_21", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8}},
      {{3, 0}, {3, 0, 192, "Kepler",  "GK10x",     2, 16,  63, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_30", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 2}, {3, 2, 192, "Kepler",  "Tegra K1",  2, 16, 255, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_32", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 5}, {3, 5, 192, "Kepler",  "GK11x",     2, 32, 255, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_35", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{3, 7}, {3, 7, 192, "Kepler",  "GK21x",    -1, -1, 255, 65536, 49152, 1024, 16, 2048, 64, 256, 131072, 256, -1, 98304, 32, "sm_37", 32,  4, -1,  -1, -1, 32, 32, 16, 16, 192, 192, 32, 32}},
      {{5, 0}, {5, 0, 128, "Maxwell", "GM10x",     2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 65536, 32, "sm_50", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{5, 2}, {5, 2, 128, "Maxwell", "GM20x",     2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 98304, 32, "sm_52", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{5, 3}, {5, 3, 256, "Maxwell", "Tegra X1",  2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 65536, 32, "sm_53", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1}},
      {{6, 0}, {6, 0,  64, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_60", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{6, 1}, {6, 1, 128, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_61", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{6, 2}, {6, 2, 128, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_62", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{7, 0}, {7, 0,   0, "Volta",   "GV10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 98304,  0, "sm_70", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}},
      {{7, 1}, {7, 1,   0, "Volta",   "GV10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 98304,  0, "sm_71", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0}}
   };

   return info.at ({cc_major, cc_minor});
}

inline auto const & get_device_info (cudaDeviceProp const & props) {
   return pfc::cuda::get_device_info (props.major, props.minor);
}

inline auto const & get_device_info (int const device = 0) {
   return pfc::cuda::get_device_info (pfc::cuda::get_device_props (device));
}

/**
 * Stringizes a CUDA-runtime-version number (as defined in CUDART_VERSION).
 */
inline auto cudart_version_to_string (int const version) noexcept {
   return std::to_string (version / 1000) + '.' + std::to_string (version % 100 / 10);
}

// -------------------------------------------------------------------------------------------------

} }   // namespace pfc::cuda

#endif   // PFC_CUDA_DEVICE_INFO_H
