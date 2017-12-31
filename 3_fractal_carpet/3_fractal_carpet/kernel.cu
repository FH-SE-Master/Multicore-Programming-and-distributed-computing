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

	// Test the device execution
	initialize_gpu();

	auto const device_info{pfc::cuda::get_device_info()};
	auto const device_props{pfc::cuda::get_device_props()};

	std::cout << "CPUs: " << CPU_COUNT << std::endl;
	std::cout << std::endl;

	// Test the host execution serial and parallel
	test_host_globally_parallel_locally_sequential(PICTURE_COUNT);
	test_host_globally_sequential_locally_parallel(PICTURE_COUNT, CPU_COUNT);

	// Test the device execution compared to serial and parallel host execution
	auto duration_thread_single_each_pic = mpv_runtime::run_with_measure(1, [&]
	{
		execute_fractal_serial_each_picture(
			PICTURE_COUNT, SIZE, MAX_ITERATIONS, DIR_GPU_TEST);
	});
	auto duration_thread_multiple_each_pic = mpv_runtime::run_with_measure(1, [&]
	{
		execute_fractal_parallel_each_picture(
			PICTURE_COUNT, PICTURE_COUNT, SIZE, MAX_ITERATIONS, DIR_GPU_TEST);
	});
	auto duration_thread_single_each_row = mpv_runtime::run_with_measure(1, [&]
	{
		execute_fractal_serial_each_rows(
			PICTURE_COUNT, PICTURE_COUNT, SIZE, MAX_ITERATIONS, DIR_GPU_TEST);
	});
	auto duration_thread_multiple_each_row = mpv_runtime::run_with_measure(1, [&]
	{
		execute_fractal_parallel_each_rows(
			PICTURE_COUNT, PICTURE_COUNT, SIZE, MAX_ITERATIONS, DIR_GPU_TEST);
	});

	auto duration_gpu = mpv_runtime::run_with_measure(1, [&]
	{
		test_gpu(PICTURE_COUNT, MAX_ITERATIONS, SIZE);
	});

	std::cout
		<< std::endl
		<< "CPU multiple thread (each pic): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single_each_pic).count() << " milliseconds"
		<< std::endl;
	std::cout
		<< std::endl
		<< "CPU multiple thread (each pic): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple_each_pic).count() << " milliseconds"
		<< std::endl;
	std::cout
		<< "Speedup thread single/multiple (each pic): "
		<< mpv_runtime::speedup(duration_thread_single_each_pic, duration_thread_multiple_each_pic)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;	
			
	std::cout
		<< std::endl
		<< "CPU multiple thread (each row): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single_each_row).count() << " milliseconds"
		<< std::endl;
	std::cout
		<< std::endl
		<< "CPU multiple thread (each row): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple_each_row).count() << " milliseconds"
		<< std::endl;
	std::cout
		<< "Speedup thread single/multiple (each row): "
		<< mpv_runtime::speedup(duration_thread_single_each_row, duration_thread_multiple_each_row)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;

	std::cout
		<< "Speedup gpu single-thread/gpuv (each pic): "
		<< mpv_runtime::speedup(duration_thread_single_each_pic, duration_gpu)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;

	std::cout
		<< "Speedup gpu multiple-thread/gpu (each pic): "
		<< mpv_runtime::speedup(duration_thread_multiple_each_pic, duration_gpu)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;

	std::cout
		<< "Speedup gpu single-thread/gpu (each row): "
		<< mpv_runtime::speedup(duration_thread_single_each_row, duration_gpu)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;

	std::cout
		<< "Speedup gpu multiple-thread/gpu (each row): "
		<< mpv_runtime::speedup(duration_thread_multiple_each_row, duration_gpu)
		<< " | tasks : " << std::to_string(PICTURE_COUNT)
		<< std::endl;
}
