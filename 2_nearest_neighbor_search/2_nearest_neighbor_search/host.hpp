#if !defined _h_host_
#define      _h_host_

#include "device_host.hpp"
#include <vector_types.h>
#include <vector>

int ceil_div(int const a, int const b);

float dice(int const range);

inline void find_all_closest_CPU(int g_points, float3* const hp_points, int* const hp_indices_h)
{
	for (size_t i = 0; i < g_points - 1; i++)

	{
		hp_indices_h[i] = find_closest(g_points, hp_points, hp_points + i);
	}
} // find_all_closest_CPU

inline dim3 grid_size(dim3 const& block, int3 const& size)
{
	dim3 grid_size{};
	grid_size.x = (block.x + size.x - 1) / block.x;
	grid_size.y = (block.y + size.y - 1) / block.y;
	grid_size.z = (block.z + size.z - 1) / block.z;

	return grid_size;
} // grid_size

inline void allocate_memory(const int g_points,
	int* & hp_indices_d,
	int* & hp_indices_h,
	int* & dp_indices,
	float3* & hp_points,
	float3* & dp_points)
{
	// host allocation
	hp_indices_d = new int[g_points] {};
	hp_indices_h = new int[g_points] {};
	hp_points = new float3[g_points]{};

	// device allocation
	mpv_exception::check(cudaMalloc(&dp_points, g_points * sizeof(float3)));
	mpv_exception::check(cudaMalloc(&dp_indices, g_points * sizeof(float3)));
} // allocate_memory

inline void free_memory(
	int* & hp_indices_d, int* & hp_indices_h,
	float3* & hp_points, int* & dp_indices,
	float3* & dp_points
)
{
	// host allocation
	hp_indices_d = nullptr;
	hp_indices_h = nullptr;
	hp_points = nullptr;

	// device allocation
	mpv_exception::check(cudaFree(dp_points));
	mpv_exception::check(cudaFree(dp_indices));
} // free_memory

// Creates the random data points 
// @param count: the count of data points
// @param lower_bound: the lower bound of the to generate value
// @param upper_bound: the upper bound of the to generate value
inline std::vector<float3> create_data(const unsigned int count, const float lower_bound, const float upper_bound)
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
#endif
