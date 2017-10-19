#include "cuda_runtime.h"
#include "pfc_cuda_device_info.h"

#include <iostream>
#include <stdio.h>
#include "util.hpp"

using namespace std::literals;

// The kernel function
__global__ void cs_kernel(int const text_size, char* const dp_destination, char* const dp_source)
{
	auto const i {blockIdx.x * blockDim.x + threadIdx.x};
	
	if(i < text_size) {
		dp_destination[i] = dp_source[i];
	}
}

int main()
{
	try
	{
		int count{0};
		cudaGetDeviceCount(&count);

		if (count > 0)
		{
			cudaSetDevice(0);
			auto const device_info = pfc::cuda::get_device_info();
			auto const device_props = pfc::cuda::get_device_props();

			std::cout << "compute capability: " << device_info.cc_major << "." << device_info.cc_minor << std::endl;
			std::cout << "Device: " << device_props.name << std::endl;

			// Makes auto to intepret variable as a string type and not a character array
			// Since C++ 11, availability to define literal type via suffix
			// {} is a initializer
			auto const text{"Hello world"s};
			auto const text_size{std::size(text) + 1};
			auto const tib{32}; // threads/block
			auto const big{(text_size + tib - 1) / tib}; // block/grid
			auto const* const hp_source{text.c_str()}; // points to start of string
			auto* hp_destination{new char[text_size]{0}}; // points to device which is empty at initialization time

			// Buffer on device
			char* dp_source{nullptr};
			char* dp_destination{ nullptr };
			mpv_exception::check(cudaMalloc(&dp_source, text_size));
			mpv_exception::check(cudaMalloc(&dp_destination, text_size));

			// copy string host -> device, expensive operation
			mpv_exception::check(cudaMemcpy(dp_source, hp_source, text_size, cudaMemcpyHostToDevice));

			// kernel call
			cs_kernel <<<big,tib>>> (text_size, dp_destination, dp_source);

			mpv_exception::check(cudaDeviceSynchronize());
			mpv_exception::check(cudaGetLastError());

			// copy result device -> host
			mpv_exception::check(cudaMemcpy(hp_destination, dp_destination, text_size, cudaMemcpyDeviceToHost));
			std::cout << "result: " << hp_destination << std::endl;
			
			// free memory on host
			delete[] hp_destination, hp_destination = nullptr;
			// free memory on device
			mpv_exception::check(cudaFree(dp_source));
			mpv_exception::check(cudaFree(dp_destination));
		}
	}
	catch (std::exception const& ex)
	{
		std::cerr << ex.what() << "\n";
	}
	return 0;
}
