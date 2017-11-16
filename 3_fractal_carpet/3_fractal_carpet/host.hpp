#if !defined __HOST__
#define      __HOST__

#include <ostream>
#include <iostream>
#include "host_device.hpp"
#include "pfc_parallel.h"

__host__ void inline calculate_fractal_host(const int size, const int max_iterations, int chunk, int begin, int end)
{
	pfc::bitmap bitmap{size,size};
	calculate_fractal(size, size, max_iterations, pfc::complex<float>(begin, end), bitmap.get_pixels());
	const auto filename = "C:/Users/S1610454013/fractal-calculate_fractal_host_" + std::to_string(chunk) + "_" + std::
		to_string(begin) + "_" +
		std::to_string(end) + ".jpg";
	bitmap.to_file(filename);
	std::cout << "Wrote result to '" << filename << "'" << std::endl;
}

__host__ void inline execute_fractal_serial(const int count, const int size, const int max_iterations)
{
	pfc::task_group task_group;
	pfc::bitmap bitmap{size, size};

	for (auto i = 0; i < count; ++i)
	{
		const auto tmp = -1 * i - 1;
		std::cout << "Calculating single threaded (" << i << ")" << std::endl << std::endl;
		calculate_fractal_host(size, max_iterations, tmp, tmp, tmp);
	}
}

__host__ void inline execute_fractal_parallel(const int count, const int size, const int max_iterations)
{
	pfc::task_group task_group{};
	std::cout << "Calculating " + std::to_string(count) + " fractal images" << std::endl;

	pfc::parallel_range(task_group, (count * 10), count, [size, max_iterations](int chunk, int begin, int end)
                    {
	                    calculate_fractal_host(size, max_iterations, chunk, begin, end);
                    });
}

__host__ void inline test_host()
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
	                                                            execute_fractal_serial(run_count, size, max_iterations);
                                                            });

	std::cout << "CPU time (single thread): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single).count() << " milliseconds" << std::
		endl << std::endl;

	auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
                                                              {
	                                                              execute_fractal_parallel(
		                                                              run_count, size, max_iterations);
                                                              });

	std::cout << "CPU time (multiple threads): "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple).count() << " milliseconds" << std
		::
		endl << std::endl;

	std::cout << "Speedup: " << mpv_runtime::speedup(duration_thread_single, duration_thread_multiple) << std::endl;

	std::cout
		<< "#################################################" << std::endl
		<< "Ended Host tests" << std::endl
		<< "#################################################" << std::endl;
}

#endif
