#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ostream>
#include <iostream>
#include <array>
#include "util.hpp"
#include "host_device.hpp"
#include "pfc_parallel.h"
#include "host.hpp"
#include "device.hpp"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

int main()
{
	try
	{
		auto count{0};
		mpv_exception::check(cudaGetDeviceCount(&count));
		if (count > 0)
		{
			cudaSetDevice(0);

			auto const device_info{pfc::cuda::get_device_info()};
			auto const device_props{pfc::cuda::get_device_props()};

			std::cout << "Device            : " << device_props.name << std::endl;
			std::cout << "Compute capability: " << device_info.cc_major << "." << device_info.cc_minor << std::endl;
			std::cout << "Arch              : " << device_info.uarch << std::endl;
			std::cout << std::endl;

			// Test the host execution serial and parallel
			test_host(15);

			// Test the device execution

		}
	}
	catch (std::exception const& x)
	{
		std::cerr << x.what() << std::endl;
	}
}
