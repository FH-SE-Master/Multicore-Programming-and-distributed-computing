//       $Id: pfc_cuda_device_info.h 35719 2017-10-01 11:58:06Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/SE/MPV3/2016-WS/ILV/src/handouts/pfc_cuda_device_info.h $
// $Revision: 35719 $
//     $Date: 2017-10-01 13:58:06 +0200 (So., 01 Okt 2017) $
//   Creator: peter.kulczycki<AT>fh-hagenberg.at
//   $Author: p20068 $
//
// Copyright: (c) 2017 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: Distributed under the Boost Software License, Version 1.0 (see
//            http://www.boost.org/LICENSE_1_0.txt).

#if !defined PFC_CUDA_DEVICE_INFO_H
#define      PFC_CUDA_DEVICE_INFO_H

//#include "./pfc_cuda_exception.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <algorithm>
#include <vector>

// -------------------------------------------------------------------------------------------------

namespace pfc {
	namespace cuda {

		struct device_info final {
			/* 0*/ int          cc_major{ 0 };    //
			/* 1*/ int          cc_minor{ 0 };    //
			/* 2*/ int          cores_sm{ 0 };    //
												  // /* 3*/ int          fp32_sm               {0};    // same as 'fp32_units_sm' ?
			/* 4*/ char const * uarch{ "" };   //
			/* 5*/ char const * chip{ "" };   //
			/* 6*/ int          ipc{ 0 };    //
			/* 7*/ int          max_act_cores_sm{ 0 };    //
			/* 8*/ int          max_regs_thread{ 0 };    //
			/* 9*/ int          max_regs_block{ 0 };    //
			/*10*/ int          max_smem_block{ 0 };    //
			/*11*/ int          max_threads_block{ 0 };    //
			/*12*/ int          max_act_blocks_sm{ 0 };    //
			/*13*/ int          max_threads_sm{ 0 };    //
			/*14*/ int          max_warps_sm{ 0 };    //
			/*15*/ int          alloc_gran_regs{ 0 };    //
			/*16*/ int          regs_sm{ 0 };    // 32-bit registers
			/*17*/ int          alloc_gran_smem{ 0 };    //
			/*18*/ int          smem_bank_width{ 0 };    //
			/*19*/ int          smem_sm{ 0 };    // [bytes]
			/*20*/ int          smem_banks{ 0 };    //
			/*21*/ char const * sm_version{ "" };   //
			/*22*/ int          warp_size{ 0 };    // [threads]
			/*23*/ int          alloc_gran_warps{ 0 };    //
			/*24*/ int          schedulers_sm{ 0 };    //
			/*25*/ int          width_cl1{ 0 };    //
			/*26*/ int          width_cl2{ 0 };    //
			/*27*/ int          load_store_units_sm{ 0 };    //
			/*28*/ int          load_store_throughput{ 0 };    // per cycle
			/*29*/ int          texture_units_sm{ 0 };    //
			/*30*/ int          texture_throughput{ 0 };    // per cycle
			/*31*/ int          fp32_units_sm{ 0 };    // same as 'fp32_sm' ?
			/*32*/ int          fp32_throughput{ 0 };    // per cycle
			/*33*/ int          sf_units_sm{ 0 };    // special function unit (sin, cosine, square root, etc.)
			/*34*/ int          sfu_throughput{ 0 };    // per cycle
		};

		// -------------------------------------------------------------------------------------------------

		inline cudaDeviceProp get_device_props(int const device = 0) {
			cudaDeviceProp props cudaDevicePropDontCare; /*PFC_CUDA_CHECK*/(cudaGetDeviceProperties(&props, device)); return props;
		}

		/**
		* see <http://en.wikipedia.org/wiki/CUDA>
		* see <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities>
		* see <https://devblogs.nvidia.com/parallelforall/inside-volta>
		*/
		inline pfc::cuda::device_info const & get_device_info(int const cc_major, int const cc_minor) {
			static std::vector <pfc::cuda::device_info> const info{
				//     0  1    2      3    4          5            6   7    8      9     10    11  12    13  14   15      16   17  18     19  20  21       22  23  24   25  26  27  28  29  30   31   32  33  34
				{ 0, 0,   0 /*,  0*/, "",        "",          0,  0,   0,     0,     0,    0,  0,    0,  0,   0,      0,   0,  0,     0,  0, "",       0,  0,  0,   0,  0,  0,  0,  0,  0,   0,   0,  0,  0 },   //

				{ 1, 0,   8 /*,  2*/, "Tesla",   "G80",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_10", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1 },   // ISA_1
				{ 1, 1,   8 /*,  2*/, "Tesla",   "G8x",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_11", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1 },   // gmem atomics
				{ 1, 2,   8 /*,  2*/, "Tesla",   "G9x",       1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_12", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1 },   // smem atomics, vote instructions
				{ 1, 3,   8 /*,  2*/, "Tesla",   "GT20x",     1,  1,  -1,    -1,    -1,   -1,  8,   -1, -1,  -1,     -1,  -1, -1,    -1, 16, "sm_13", -1, -1,  1, 128, 32, -1, -1, -1, -1,   2,  -1, -1, -1 },   // double precision floating-point support

				{ 2, 0,  32 /*,  4*/, "Fermi",   "GF10x",     1, 16,  63, 32768, 49152, 1024,  8, 1536, 48,  64,  32768, 128,  4, 49152, 32, "sm_20", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8 },   // Fermi support
				{ 2, 1,  48 /*,  8*/, "Fermi",   "GF10x",     2, 16,  63, 32768, 49152, 1024,  8, 1536, 48,  64,  32768, 128,  4, 49152, 32, "sm_21", 32,  2,  2, 128, 32, 16, 16,  4,  4,  32,  64,  4,  8 },   // more cores

				{ 3, 0, 192 /*, 32*/, "Kepler",  "GK10x",     2, 16,  63, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_30", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32 },   // Kepler support
				{ 3, 2, 192 /*, 32*/, "Kepler",  "Tegra K1",  2, 16, 255, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_32", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32 },   //
				{ 3, 5, 192 /*, 32*/, "Kepler",  "GK11x",     2, 32, 255, 65536, 49152, 1024, 16, 2048, 64, 256,  65536, 256,  4, 49152, 32, "sm_35", 32,  4,  4, 128, 32, 32, 32, 16, 16, 192, 192, 32, 32 },   // dynamic parallelism
				{ 3, 7, 192 /*, -1*/, "Kepler",  "GK21x",    -1, -1, 255, 65536, 49152, 1024, 16, 2048, 64, 256, 131072, 256, -1, 98304, 32, "sm_37", 32,  4, -1,  -1, -1, 32, 32, 16, 16, 192, 192, 32, 32 },   //

				{ 5, 0, 128 /*, 32*/, "Maxwell", "GM10x",     2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 65536, 32, "sm_50", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1 },   // Maxwell support
				{ 5, 2, 128 /*, 32*/, "Maxwell", "GM20x",     2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 98304, 32, "sm_52", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1 },   //
				{ 5, 3, 256 /*, 32*/, "Maxwell", "Tegra X1",  2, 32, 255, 32768, 49152, 1024, 32, 2048, 64, 256,  65536, 256,  4, 65536, 32, "sm_53", 32,  4,  4, 128, 32, -1, -1, -1, -1, 128,  -1, -1, -1 },   //

				{ 6, 0,  64 /*,  0*/, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_60", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0 },   // Pascal support
				{ 6, 1, 128 /*,  0*/, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_61", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0 },   //
				{ 6, 2, 128 /*,  0*/, "Pascal",  "GP10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 65536,  0, "sm_62", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0 },   //

				{ 7, 0,   0 /*,  0*/, "Volta",   "GV10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 98304,  0, "sm_70", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0 },   // Volta support
				{ 7, 1,   0 /*,  0*/, "Volta",   "GV10x",     0,  0, 255, 65536,     0, 1024, 32, 2048, 64,   0,  65536,   0,  0, 98304,  0, "sm_71", 32,  0,  0,   0,  0,  0,  0,  0,  0,  64,   0,  0,  0 },   //
			};

			for (auto const & e : info) {
				if ((e.cc_major == cc_major) && (e.cc_minor == cc_minor)) {
					return e;
				}
			}

			return info[0];
		}

		inline pfc::cuda::device_info const & get_device_info(cudaDeviceProp const & props) {
			return pfc::cuda::get_device_info(props.major, props.minor);
		}

		inline pfc::cuda::device_info const & get_device_info(int const device = 0) {
			return pfc::cuda::get_device_info(pfc::cuda::get_device_props(device));
		}

		inline std::string version_to_string(int const version) {
			return std::to_string(version / 1000) + '.' + std::to_string(version % 100 / 10);
		}

	}
}   // namespace cuda, namespace pfc

	// -------------------------------------------------------------------------------------------------

#endif   // PFC_CUDA_DEVICE_INFO_H