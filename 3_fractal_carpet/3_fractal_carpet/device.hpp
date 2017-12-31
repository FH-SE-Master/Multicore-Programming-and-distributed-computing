#if !defined __DEVICE__
#define      __DEVICE__
#include "host_device.hpp"
#include "pfc_cuda_device_info.h"
#include "util.hpp"
#include "pfc_cuda_memory.h"
#include "pfc_cuda_exception.h"
#include "pfc_parallel.h"
#include "constant.h"

CATTR_KERNEL void fractal_kernel(pfc::complex<float> start,
	const int maxIterations,
	const int size,
	pfc::bitmap::pixel_t * result,
	pfc::bitmap::pixel_t* rgb_map) {

	auto row{ blockIdx.x*blockDim.x + threadIdx.x };
	auto col{ blockIdx.y*blockDim.y + threadIdx.y };

	calculate_fractal_part(size, maxIterations, row, col, start, result, rgb_map);
}

void initialize_gpu() {
	int count{ 0 }; PFC_CUDA_CHECK(cudaGetDeviceCount(&count));
	if (count > 0) {
		cudaSetDevice(0);

		auto const deviceInfo{ pfc::cuda::get_device_info() };
		auto const deviceProps{ pfc::cuda::get_device_props() };

		std::cout << "Device name       : " << deviceProps.name << std::endl;
		std::cout << "Cmpute Capability : " << deviceInfo.cc_major << "." << deviceInfo.cc_minor << std::endl;
		std::cout << "Arch              : " << deviceInfo.uarch << std::endl;
		std::cout << std::endl;
	}
}

CATTR_HOST void inline test_gpu(const int pictureCount, 
	const int size,
	const int maxIterations) {
	try {
		pfc::bitmap::pixel_t* device_pixels{ CUDA_MALLOC(pfc::bitmap::pixel_t, size*size) };
		pfc::bitmap::pixel_t* device_rgb_map{ CUDA_MALLOC(pfc::bitmap::pixel_t, 16) };

		CUDA_MEMCPY(device_rgb_map, RGB_MAPPING, RGB_COLOR_SIZE, cudaMemcpyHostToDevice);
		
		for (int i = 0; i < pictureCount; ++i) {
			pfc::bitmap bitmap(size, size);
			fractal_kernel << < gpu_grid_size, gpu_block_size >> > (pfc::complex<float>(0, 0), maxIterations, size, device_pixels, device_rgb_map);
			PFC_CUDA_CHECK(cudaGetLastError());
			PFC_CUDA_CHECK(cudaDeviceSynchronize()); // synchronize with device, means wait for it
			PFC_CUDA_CHECK(cudaGetLastError());
			PFC_CUDA_MEMCPY(bitmap.get_pixels(), device_pixels, size*size, cudaMemcpyDeviceToHost);

			bitmap.to_file(DIR_GPU_TEST + "fractal-gpu_" + std::to_string(i) + ".jpg");
		}

		CUDA_FREE(device_pixels);
	}
	catch (std::exception const &x) {
		std::cerr << x.what() << std::endl;
	}
}
#endif
