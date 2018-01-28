#include <stdio.h>
#include <ostream>
#include <iostream>
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.hpp"
#include "pfc_cuda_memory.h"
#include "pfc_cuda_exception.h"
#include "pfc_parallel.h"
#include "device.hpp"
#include "host.hpp"

CATTR_KERNEL void fractal_kernel(pfc::complex<double> start,
	const int maxIterations,
	const int size,
	pfc::bitmap::pixel_t * result,
	pfc::bitmap::pixel_t* rgb_map) {

	auto row{ blockIdx.x*blockDim.x + threadIdx.x };
	auto col{ blockIdx.y*blockDim.y + threadIdx.y };

	calculate_fractal_part(size, maxIterations, row, col, start, result, rgb_map);
}

CATTR_HOST void initialize_gpu() {
	int count{ 0 }; PFC_CUDA_CHECK(cudaGetDeviceCount(&count));
	if (count > 0) {
		cudaSetDevice(0);

		auto const deviceInfo{ pfc::cuda::get_device_info() };
		auto const deviceProps{ pfc::cuda::get_device_props() };

		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << "Device Metadata:" << std::endl;
		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << "Device: " << deviceProps.name << std::endl;
		std::cout << "Compute capability: " << deviceInfo.cc_major << "." << deviceInfo.cc_minor << std::endl;
		std::cout << "Arch: " << deviceInfo.uarch << std::endl;
		std::cout << "Cores: " << deviceInfo.cores_sm << std::endl;
		std::cout << "Global memory: " << deviceProps.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
		std::cout << "Shared memory per block: " << deviceProps.sharedMemPerBlock / 1024 << " KB" << std::endl;
		std::cout << "Shared memory per multiprocessor: " << deviceProps.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
		std::cout << "Max threads per block: " << deviceInfo.max_threads_block << std::endl;
		std::cout << "-----------------------------------------------------" << std::endl;
		std::cout << std::endl;
	}
}

CATTR_HOST void inline execute_gpu_global_parallel_local_serial(const int pictureCount, 
	const int size,
	const int maxIterations,
	dim3 block_size) {
	try {

		pfc::bitmap::pixel_t* device_pixels{ CUDA_MALLOC(pfc::bitmap::pixel_t, size*size) };
		pfc::bitmap::pixel_t* device_rgb_map{ CUDA_MALLOC(pfc::bitmap::pixel_t, 16) };

		CUDA_MEMCPY(device_rgb_map, RGB_MAPPING, RGB_COLOR_SIZE, cudaMemcpyHostToDevice);
		
		for (int i = 0; i < pictureCount; ++i) {
			dim3 grid_size((size + block_size.x - 1) / block_size.x, (size + block_size.y - 1) / block_size.y);
			pfc::bitmap bitmap(size, size);
			fractal_kernel << < grid_size, block_size >> > (pfc::complex<double>(0, 0), maxIterations, size, device_pixels, device_rgb_map);
			PFC_CUDA_CHECK(cudaGetLastError());
			PFC_CUDA_CHECK(cudaDeviceSynchronize()); // synchronize with device, means wait for it
			PFC_CUDA_CHECK(cudaGetLastError());
			PFC_CUDA_MEMCPY(bitmap.get_pixels(), device_pixels, size*size, cudaMemcpyDeviceToHost);

			//bitmap.to_file(DIR_GPU_TEST + "fractal-gpu_" + std::to_string(block_size.x) + "_" + std::to_string(block_size.y) + "_" + std::to_string(i) + ".jpg");
		}

		CUDA_FREE(device_pixels);
	}
	catch (std::exception const &x) {
		std::cerr << x.what() << std::endl;
	}
}
CATTR_HOST void inline test_gpu_global_parallel_local_serial() {

	std::cout
		<< "#################################################################################" << std::endl
		<< "Start GPU tests 'GPLS'" << std::endl
		<< "#################################################################################" << std::endl;

	for each (unsigned int task_count in TASK_COUNTS)
	{

		auto duration_gpu = mpv_runtime::run_with_measure(1, [&]
		{
			execute_gpu_global_parallel_local_serial(PICTURE_COUNT, MAX_ITERATIONS, SIZE, dim3{ task_count ,task_count });
		});

		display_results("GPU-GPLS", task_count, duration_gpu);
	}

	std::cout
		<< "#################################################" << std::endl
		<< "Ended GPU tests 'GPLS'" << std::endl
		<< "#################################################" << std::endl;
}

int main()
{
	initialize_gpu();

	std::cout << std::endl;

	test_gpu_global_parallel_local_serial();
}


