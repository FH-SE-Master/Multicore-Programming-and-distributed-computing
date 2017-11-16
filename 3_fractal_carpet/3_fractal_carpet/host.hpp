#if !defined __HOST__
#define      __HOST__

#include <ostream>
#include <iostream>
#include "host_device.hpp"
#include "pfc_parallel.h"

__host__ void inline calculate_fractal_host(const int size, const int max_iterations, int chunk, int begin, int end)
{
	pfc::bitmap bitmap{size,size};
	calculate_fractal(size, size, max_iterations, begin, end, pfc::complex<float>(begin, end), bitmap.get_pixels());
	const auto filename = "C:/Users/S1610454013/fractal-calculate_fractal_host_" + std::to_string(chunk) + "_" + std::
		to_string(begin) + "_" +
		std::to_string(end) + ".jpg";
	bitmap.to_file(filename);
	std::cout << "Wrote result to '" << filename << "'" << std::endl;
}

__host__ void inline execute_fractal_serial_each_picture(const int count, const int size, const int max_iterations)
{
	pfc::task_group task_group;
	pfc::bitmap bitmap{size, size};

	for (auto i = 0; i < count; ++i)
	{
		const auto tmp = -1 * i - 1;
		std::cout << "Calculating single thread (" << i << ")" << std::endl;
		calculate_fractal_host(size, max_iterations, tmp, 0, size);
	}
}

__host__ void inline execute_fractal_serial_each_row(const int count, const int size, const int max_iterations)
{
	pfc::task_group task_group;
	pfc::bitmap bitmap{ size, size };

	for (auto i = 0; i < count; ++i)
	{
		const auto tmp = -1 * i - 1;
		std::cout << "Calculating single thread (" << i << ")" << std::endl;
		calculate_fractal_host(size, max_iterations, tmp, 0, size);
	}
}

__host__ void inline execute_fractal_parallel(const int task_count, const int count, const int size,
                                              const int max_iterations)
{
	pfc::task_group task_group{};
	std::cout << "Calculating multiple tasks (" << std::to_string(task_count) << ")" << std::endl;

	pfc::parallel_range(task_group, task_count, count, [size, max_iterations](int chunk, int begin, int end)
                    {
	                    calculate_fractal_host(size, max_iterations, chunk, begin, end);
                    });
}

__host__ void inline test_host(const int iteration_count)
{
	const auto size = 1000;
	const auto max_iterations = 1000;
	const auto run_count = 4;

	std::cout
		<< "#################################################" << std::endl
		<< "Start Host tests" << std::endl
		<< "#################################################" << std::endl;

	auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
                                                            {
	                                                            execute_fractal_serial_each_picture(run_count, size, max_iterations);
                                                            });

	std::cout << "CPU time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single).count() << " milliseconds" << std::
		endl << std::endl;

	for (auto i = 1; i <= iteration_count; ++i)
	{
		const auto thread_count = (i * run_count);
		auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
	                                                              {
		                                                              execute_fractal_parallel(
			                                                              thread_count, run_count, size, max_iterations);
	                                                              });

		std::cout
			<< std::endl
			<< "CPU time: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple).count() << " milliseconds"
			<< std::endl;

		std::cout
			<< "Speedup: "
			<< mpv_runtime::speedup(duration_thread_single, duration_thread_multiple)
			<< " | tasks : " << std::to_string(thread_count)
			<< std::endl;

		std::cout
			<< "#################################################" << std::endl
			<< "Ended Host tests" << std::endl
			<< "#################################################" << std::endl;
	}
}

#endif
