#if !defined _h_device_host_
#define      _h_device_host_
#include <device_launch_parameters.h>

__constant__ auto const g_block_size = 128;
__constant__ auto const g_points = 50000;

__host__ __device__ __forceinline__
float norm(float3 const& p, float3 const& q)
{
	const auto x_diff = p.x - q.x;
	const auto y_diff = p.y - q.y;
	const auto z_diff = p.z - q.z;

	return (x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff);
}

__host__ __device__ __forceinline__
int find_closest(const int g_points, float3* p_points, float3 const* p_point)
{
	int index = -1;
	auto min_so_far = FLT_MAX;

	for (auto to = 0; to < g_points; ++p_points, ++to)
	{
		if (p_points != p_point)
		{
			auto const dist = norm(*p_point, *p_points);

			if (dist < min_so_far)
			{
				min_so_far = dist;
				index = to;
			}
		}
	}

	return index;
}


#endif
