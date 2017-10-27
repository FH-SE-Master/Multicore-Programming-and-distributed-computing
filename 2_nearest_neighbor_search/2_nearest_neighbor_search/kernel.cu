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

auto const cpu_count = std::max<int>(1, std::thread::hardware_concurrency());
auto const run_count = 3;

__constant__ auto const g_block_size = 64;
__constant__ auto const g_points = 75000;
auto const g_grid_size = grid_size(
	g_block_size, {g_points , 1, 1}
);

auto const point_count = 100;

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

			auto const tib{g_block_size}; //threads in block
			auto const big{(g_points + tib - 1) / tib}; //blocks in grid + round
			auto hp_points{std::make_unique<float3[]>(g_points)};
			auto hp_result{std::make_unique<int[]>(g_points)};

			// create data points
			create_data(g_points, hp_points.get(), 1, 10000);

			// Allocate on device
			std::cout << "Allocating memory on device ..." << std::endl;
			float3* dp_points = CUDA_MALLOC(float3, g_points);
			int* dp_result = CUDA_MALLOC(int, g_points);

			std::cout << "Calculating distances on device (block size " << g_block_size << ", " << run_count << " runs) ..." <<
				std::endl << std::endl;
			auto const duration_gpu = mpv_runtime::run_with_measure(run_count, [&]
		                                                        {
			                                                        CUDA_MEMCPY(dp_points, hp_points.get(), g_points,
				                                                        cudaMemcpyHostToDevice);
			                                                       find_all_closest_GPU << <big, tib >> >(
				                                                        g_points, dp_points, dp_result);
			                                                        cudaDeviceSynchronize();
			                                                        mpv_exception::check(cudaGetLastError());
			                                                        CUDA_MEMCPY(hp_result.get(), dp_result, g_points,
				                                                        cudaMemcpyDeviceToHost);
		                                                        });

			std::cout << "GPU time (average of " << run_count << " runs): "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_gpu).count() << " milliseconds" << std::endl
				<< std::endl;

			std::cout << "Warming up CPU ..." << std::endl << std::endl;
			mpv_threading::warm_up_cpu(5s);

			std::cout << "Calculating distances on host (" << cpu_count << " threads, " << run_count << " runs ) ..." << std::
				endl << std::endl;

			auto chunk{(g_points + cpu_count - 1) / cpu_count};
			auto const duration_cpu = mpv_runtime::run_with_measure(run_count, [&]
		                                                        {
			                                                        std::vector<std::future<void>> task_group;
			                                                        for (auto i = 0; i < cpu_count; i++)
			                                                        {
				                                                        auto index{i};
				                                                        task_group.push_back(std::async(std::launch::async, [&]
			                                                                                        {
				                                                                                        find_all_closest_CPU(
					                                                                                        g_points, hp_points.get(),
					                                                                                        hp_result.get(),
					                                                                                        std::make_pair(
						                                                                                        index * chunk,
						                                                                                        (index + 1) * chunk));
			                                                                                        }));
			                                                        }
			                                                        for (auto& f : task_group)
			                                                        {
				                                                        f.get();
			                                                        }
		                                                        });

			std::cout
				<< "CPU time (average of " << run_count << " runs): "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_cpu).count() << " milliseconds" << std::endl <<
				std::endl;
			std::cout << "Speedup: " << mpv_runtime::speedup(duration_cpu, duration_gpu) << std::endl;
		}
	}
	catch (std::exception const& x)
	{
		std::cerr << x.what() << std::endl;
	}
}
