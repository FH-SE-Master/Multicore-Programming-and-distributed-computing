#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pfc_cuda_device_info.h"
#include "util.hpp"
#include "host.hpp"
#include "device_host.hpp"

#include <vector_types.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <complex>

using namespace std::literals;

auto const g_grid_size = grid_size(
	g_block_size, {g_points , 1, 1}
);

auto const point_count = 100;

int main()
{
	try
	{
		auto count{0};
		cudaGetDeviceCount(&count);
		cudaSetDevice(0);

		if (count > 0)
		{
			mpv_device::print_device_info();
			std::vector<float3> data = create_data(point_count, 0.0, 100.0);

			auto const tib{32}; // threads/block
			auto const big{(point_count * sizeof(float3) + tib - 1) / tib};
			int* hp_indices_d = nullptr;
			int* hp_indices_h = nullptr;
			int* dp_indices = nullptr;
			float3* hp_points = nullptr;
			float3* dp_points = nullptr;
			std::copy(data.begin(), data.end(), hp_points);
			std::copy(data.begin(), data.end(), dp_points);

			allocate_memory(point_count, hp_indices_d, hp_indices_h, dp_indices, hp_points, dp_points);
		}
	}
	catch (std::runtime_error)
	{
	}
}
