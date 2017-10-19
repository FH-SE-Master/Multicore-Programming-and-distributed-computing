#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ostream>
#include <iostream>
#include "util.hpp"
#include "host_device.hpp"

const int HEIGHT = 1000;
const int WIDTH = HEIGHT;
const int MAX_ITERATIONS = 10000;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void fractal_kernel(int* c, const int* a, const int* b)
{
}

int main()
{
	try
	{
		int count{0};
		mpv_exception::check(cudaGetDeviceCount(&count));
		if (count > 0)
		{
			cudaSetDevice(0);

			auto const deviceInfo{pfc::cuda::get_device_info()};
			auto const deviceProps{pfc::cuda::get_device_props()};

			std::cout << "Device            : " << deviceProps.name << std::endl;
			std::cout << "Compute capability: " << deviceInfo.cc_major << "." << deviceInfo.cc_minor << std::endl;
			std::cout << "Arch              : " << deviceInfo.uarch << std::endl;
			std::cout << std::endl;

			std::cout << "Calculating single threaded" << std::endl << std::endl;
			pfc::bitmap bitmap{WIDTH, HEIGHT};
			auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
		                                                            {
			                                                            calculate_fractal(HEIGHT, WIDTH, MAX_ITERATIONS, bitmap.get_pixels(), std::pair<int, int>{0,0});
		                                                            });
			bitmap.to_file("fractal-0-0.jpg");
			std::cout << "CPU time (single thread): "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single).count() << " milliseconds" << std::
				endl << std::endl;

			
		}
	}
	catch (std::exception const& x)
	{
		std::cerr << x.what() << std::endl;
	}
}
