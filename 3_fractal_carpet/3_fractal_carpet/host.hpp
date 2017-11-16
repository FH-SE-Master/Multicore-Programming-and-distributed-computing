#if !defined __HOST__
#define      __HOST__

#include <ostream>
#include <iostream>
#include "host_device.hpp"
#include "pfc_parallel.h"

__host__ void inline execute_fractal_serial_each_picture(const int picture_count, const int size,
                                                         const int max_iterations)
{
	for (auto i = 1; i <= picture_count; ++i)
	{
		pfc::bitmap bitmap{size, size};
		std::cout << "Calculating single thread (" << i << ")" << std::endl;
		calculate_fractal(size, max_iterations, 0, size, pfc::complex<float>(0, 0), bitmap.get_pixels());
		const auto filename = "C:/Users/S1610454013/fractal-execute_fractal_serial_each_picture_" + std::to_string(i) +
			".jpg";
		bitmap.to_file(filename);
		std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}

__host__ void inline execute_fractal_serial_each_rows(const int picture_count, const int task_count, const int size,
                                                      const int max_iterations)
{
	pfc::bitmap bitmap{size, size};
	auto part_size = size / task_count;
	auto rest_size = size - (part_size * task_count);

	for (auto i = 1; i <= picture_count; ++i)
	{
		auto start_row = 0;
		auto end_row = 0;
		for (auto i = 0; i < task_count; ++i)
		{
			end_row = start_row + part_size;
			if (i == task_count)
			{
				end_row += rest_size;
			}
			std::cout << "Calculating single thread (" << i << ")" << std::endl;
			calculate_fractal(size, max_iterations, start_row, end_row, pfc::complex<float>(0, 0), bitmap.get_pixels());
			start_row = end_row;
		}
		const auto filename = "C:/Users/S1610454013/fractal-execute_fractal_serial_each_rows" + std::to_string(i) + ".jpg";
		bitmap.to_file(filename);
		std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}

__host__ void inline execute_fractal_parallel_each_picture(const int task_count,
                                                           const int picture_count,
                                                           const int size,
                                                           const int max_iterations)
{
	std::cout << "Calculating multiple on multiple picture (" << std::to_string(task_count) << ")" << std::endl;
	pfc::task_group task_group{};

	pfc::parallel_range(task_group, task_count, picture_count, [size, max_iterations](int chunk, int begin, int end)
                    {
	                    pfc::bitmap bitmap{size, size};
	                    calculate_fractal(size, max_iterations, 0, size, pfc::complex<float>(0, 0),
	                                      bitmap.get_pixels());
	                    const auto filename = "C:/Users/S1610454013/fractal-execute_fractal_parallel_each_picture_" + std::
		                    to_string(chunk) + "_" + std::to_string(begin) + "_" + std::to_string(chunk) + ".jpg";
	                    bitmap.to_file(filename);
                    });
}

__host__ void inline execute_fractal_parallel_each_rows(const int task_count,
                                                        const int picture_count,
                                                        const int size,
                                                        const int max_iterations)
{
	std::cout << "Calculating multiple on single picture (" << std::to_string(task_count) << ")" << std::endl;
	pfc::task_group task_group{};
	std::vector<pfc::bitmap> bitmaps{};
	for (int i = 0; i < picture_count; ++i)
	{
		bitmaps.push_back(pfc::bitmap{size, size});
	}


	for (int i = 0; i < picture_count; ++i)
	{
		pfc::parallel_range(task_group, task_count, size,
		                    [size, max_iterations, task_count, &bitmaps, i](int chunk, int begin, int end)
	                    {
		                    calculate_fractal(size, max_iterations, begin, end, pfc::complex<float>(0, 0),
		                                      bitmaps[i].get_pixels());
	                    });
	}

	task_group.join_all();
	for (int i = 0; i < bitmaps.size(); ++i)
	{
		const auto filename = "C:/Users/S1610454013/fractal-execute_fractal_parallel_each_rows" + std::to_string(i) + ".jpg";
		bitmaps[i].to_file(filename);
		std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}

__host__ void inline test_host_globally_parallel_locally_sequential(const int picture_count)

{
	const auto size = 1000;
	const auto max_iterations = 1000;

	std::cout
		<< "#################################################################################" << std::endl
		<< "Start Host tests 'test_host_globally_parallel_locally_sequential'" << std::endl
		<< "#################################################################################" << std::endl;

	auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
                                                            {
	                                                            execute_fractal_serial_each_picture(
		                                                            picture_count, size, max_iterations);
                                                            });

	std::cout << "CPU time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single).count() << " milliseconds" << std::
		endl << std::endl;

	auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
                                                              {
	                                                              execute_fractal_parallel_each_picture(
		                                                              picture_count, picture_count, size, max_iterations);
                                                              });

	std::cout
		<< std::endl
		<< "CPU time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple).count() << " milliseconds"
		<< std::endl;

	std::cout
		<< "Speedup: "
		<< mpv_runtime::speedup(duration_thread_single, duration_thread_multiple)
		<< " | tasks : " << std::to_string(picture_count)
		<< std::endl;

	std::cout
		<< "#################################################################################" << std::endl
		<< "Ended Host tests" << std::endl
		<< "#################################################################################" << std::endl;
}

__host__ void inline test_host_globally_sequential_locally_parallel(const int picture_count, const int task_count)

{
	const auto size = 1000;
	const auto max_iterations = 1000;

	std::cout
		<< "#################################################################################" << std::endl
		<< "Start Host tests 'test_host_globally_sequential_locally_parallel'" << std::endl
		<< "#################################################################################" << std::endl;

	auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
                                                            {
	                                                            execute_fractal_serial_each_rows(
		                                                            picture_count, task_count, size, max_iterations);
                                                            });

	std::cout << "CPU time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_single).count() << " milliseconds" << std::
		endl << std::endl;

	auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
                                                              {
	                                                              execute_fractal_parallel_each_rows(
		                                                              task_count, picture_count, size, max_iterations);
                                                              });

	std::cout
		<< std::endl
		<< "CPU time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(duration_thread_multiple).count() << " milliseconds"
		<< std::endl;

	std::cout
		<< "Speedup: "
		<< mpv_runtime::speedup(duration_thread_single, duration_thread_multiple)
		<< " | tasks : " << std::to_string(picture_count)
		<< std::endl;

	std::cout
		<< "#################################################" << std::endl
		<< "Ended Host tests" << std::endl
		<< "#################################################" << std::endl;
}
#endif
