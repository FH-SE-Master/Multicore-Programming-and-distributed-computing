#include "util.hpp"
#include "device_host.hpp"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>



dim3 grid_size(dim3 const& block, int3 const& size)
{
	dim3 grid_size{};
	grid_size.x = (block.x + size.x - 1) / block.x;
	grid_size.y = (block.y + size.y - 1) / block.y;
	grid_size.z = (block.z + size.z - 1) / block.z;

	return grid_size;
} // grid_size

void find_all_closest_CPU(int g_points, float3* const hp_points, int* const hp_indices_h)
{
	for (size_t i = 0; i < g_points - 1; i++)

	{
		hp_indices_h[i] = find_closest(g_points, hp_points, hp_points + i);
	}
} // find_all_closest_CPU

std::vector<float3> create_data(const unsigned int count, const float lower_bound, const float upper_bound)
{
	// Check for invalid count
	if (count <= 0)
	{
		throw std::exception("Count must be > 0");
	}
	std::vector<float3> data;

	for (unsigned int i = 0; i < count; ++i)
	{
		float3 point;
		point.x = mpv_random::get_random_uniformed(lower_bound, upper_bound);
		point.y = mpv_random::get_random_uniformed(lower_bound, upper_bound);
		point.z = mpv_random::get_random_uniformed(lower_bound, upper_bound);
		data.push_back(point);
	}

	return data;
} // create_data
