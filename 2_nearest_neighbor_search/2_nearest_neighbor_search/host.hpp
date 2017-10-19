#if !defined _h_host_
#define      _h_host_

#include "device_host.hpp"
#include <vector_types.h>
#include <vector>

int ceil_div(int const a, int const b);

float dice(int const range);

inline void find_all_closest_CPU(int const  g_points, float3* const hp_points, int* const hp_result, const std::pair<int, int> part)
{
	for (int i = part.first; i < part.second; i++)

	{
		hp_result[i] = find_closest(g_points, hp_points, hp_points + i);
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

// Creates the random data points 
// @param count: the count of points to generate
// @param points: the points array to create points for
// @param lower_bound: the lower bound of the to generate value
// @param upper_bound: the upper bound of the to generate value
inline void create_data(const unsigned int count, float3* points, const float lower_bound, const float upper_bound)
{
	for (unsigned int i = 0; i < count; ++i)
	{
		points[i].x = mpv_random::get_random_uniformed(lower_bound, upper_bound);
		points[i].y = mpv_random::get_random_uniformed(lower_bound, upper_bound);
		points[i].z = mpv_random::get_random_uniformed(lower_bound, upper_bound);
	}
} // create_data
#endif
