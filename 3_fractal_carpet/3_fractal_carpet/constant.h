#if !defined _hpp_constant_
#define      _hpp_constant_
#include "pfc_cuda_macros.h"
#include <device_launch_parameters.h>
#include <array>
#include "pfc_bitmap.h"
#include "pfc_complex.h"

CATTR_CONST const auto PICTURE_COUNT{ 25 };
CATTR_CONST const auto MAX_ITERATIONS{ 100 };
CATTR_CONST const auto SIZE{ 500000 };
CATTR_CONST auto const RGB_COLOR_SIZE { 16 };

CATTR_CONST pfc::bitmap::pixel_t RGB_MAPPING[RGB_COLOR_SIZE] {
	pfc::bitmap::pixel_t{ 15, 30, 66, 0 } , pfc::bitmap::pixel_t{ 26, 7, 25, 0 } , pfc::bitmap::pixel_t{ 47, 1, 9, 0 } , pfc::bitmap::pixel_t{ 73, 4, 4, 0 } ,
	pfc::bitmap::pixel_t{ 100, 7, 0, 0 } , pfc::bitmap::pixel_t{ 138, 44, 12, 0 } , pfc::bitmap::pixel_t{ 177, 82, 24, 0 } ,
	pfc::bitmap::pixel_t{ 209, 125, 57, 0 } , pfc::bitmap::pixel_t{ 229, 181, 134, 0 } , pfc::bitmap::pixel_t{ 248, 236, 211, 0 } ,
	pfc::bitmap::pixel_t{ 191, 233, 241, 0 } , pfc::bitmap::pixel_t{ 95, 201, 248, 0 } ,pfc::bitmap::pixel_t{ 0, 170, 255, 0 } ,
	pfc::bitmap::pixel_t{ 0, 128, 204, 0 } ,pfc::bitmap::pixel_t{ 0, 87, 153, 0 } , pfc::bitmap::pixel_t{ 3, 52, 106, 0 }
};

dim3 build_gpu_grid_size(dim3 const & block, int3 const & size) {
	dim3 s;
	s.x = (size.x + block.x - 1) / block.x;
	s.y = (size.y + block.y - 1) / block.y;
	s.z = (size.z + block.z - 1) / block.z;
	return s;
}

auto const gpu_block_size = dim3{ 16,16 };

auto const gpu_grid_size = build_gpu_grid_size(
	gpu_block_size, { 5000 , 5000, 1 }
);

const std::string DIR_GPU_TEST = "gpu-test/";
const std::string DIR_CPU_TEST = "cpu-test/";
const std::string FILE_HOST_RESULT = "result.csv";
int const TASK_COUNTS[]{ 2,4,8,16,32,64,128,256,512,1024 };

#endif