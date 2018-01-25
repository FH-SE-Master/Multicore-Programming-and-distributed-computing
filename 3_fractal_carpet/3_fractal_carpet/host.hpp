#if !defined __HOST__
#define      __HOST__

#include <ostream>
#include <iostream>
#include "fractal.hpp"
#include "pfc_parallel.h"

CATTR_HOST void prepare_host_file() {
	std::ofstream file;
	file.open(FILE_HOST_RESULT, std::ios::trunc);
	file << "Name;Tasks;Ref;Target;SpeedUp" << std::endl;
	file.close();
}

template <typename duration_t>
CATTR_HOST void save_and_display_host_results(std::string name, int tasks, duration_t duration, duration_t ref) {
	auto const speedup = mpv_runtime::speedup(ref, duration);
	auto const refMillis = std::chrono::duration_cast<std::chrono::milliseconds>(ref).count();
	auto const targetMillis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	std::cout << "name: " << name << " | tasks: " << tasks << " | ref: " << refMillis << " | target: " << targetMillis << " | speedup: " << speedup << std::endl;

	std::ofstream datafile;
	datafile.open(FILE_HOST_RESULT, std::ios::out | std::ios::app);
	datafile << name << ";" << tasks << ";" << refMillis << ";" << targetMillis << ";" << speedup << std::endl;
	datafile.close();
}

CATTR_HOST void inline execute_fractal_serial_each_picture(const int picture_count, const int size,
                                                         const int max_iterations, const std::string prefix = "")
{
	for (auto i = 1; i <= picture_count; ++i)
	{
		pfc::bitmap bitmap{size, size};
		//std::cout << "Calculating single thread (picture-" << i << ")" << std::endl;
		calculate_fractal(size, max_iterations, 0, size, pfc::complex<float>(0, 0), bitmap.get_pixels(), RGB_MAPPING);
		const auto filename = prefix + "fractal-execute_fractal_serial_each_picture_" + std::to_string(i) +
			".jpg";
		bitmap.to_file(filename);
		//std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}

CATTR_HOST void inline execute_fractal_serial_each_rows(const int picture_count, const int task_count, const int size,
                                                      const int max_iterations, const std::string prefix = "")
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
			//std::cout << "Calculating single thread (part-" << i << ")" << std::endl;
			calculate_fractal(size, max_iterations, start_row, end_row, pfc::complex<float>(0, 0), bitmap.get_pixels(), RGB_MAPPING);
			start_row = end_row;
		}
		const auto filename = prefix + "fractal-execute_fractal_serial_each_rows_" + std::to_string(i) + ".jpg";
		bitmap.to_file(filename);
		//std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}

CATTR_HOST void inline execute_fractal_parallel_each_picture(const int task_count,
                                                           const int picture_count,
                                                           const int size,
                                                           const int max_iterations, 
	const std::string prefix = "")
{
	//std::cout << "Calculating multiple on multiple picture (" << std::to_string(task_count) << ")" << std::endl;
	pfc::task_group task_group{};

	pfc::parallel_range(task_group, task_count, picture_count, [size, prefix, max_iterations](int chunk, int begin, int end)
                    {
	                    pfc::bitmap bitmap{size, size};
	                    calculate_fractal(size, max_iterations, 0, size, pfc::complex<float>(0, 0),
	                                      bitmap.get_pixels(), RGB_MAPPING);
	                    const auto filename = prefix + "fractal-execute_fractal_parallel_each_picture_" + std::
		                    to_string(chunk) + "_" + std::to_string(begin) + "_" + std::to_string(chunk) + ".jpg";
	                    bitmap.to_file(filename);
						//std::cout << "Wrote result to '" << filename << "'" << std::endl;
                    });
}

CATTR_HOST void inline execute_fractal_parallel_each_rows(const int task_count,
                                                        const int picture_count,
                                                        const int size,
                                                        const int max_iterations,
	const std::string prefix = "")
{
	//std::cout << "Calculating multiple on single picture (" << std::to_string(task_count) << ")" << std::endl;
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
		                                      bitmaps[i].get_pixels(), RGB_MAPPING);
	                    });
	}

	task_group.join_all();
	for (int i = 0; i < bitmaps.size(); ++i)
	{
		const auto filename = prefix + "fractal-execute_fractal_parallel_each_rows" + std::to_string(i) + ".jpg";
		bitmaps[i].to_file(filename);
		//std::cout << "Wrote result to '" << filename << "'" << std::endl;
	}
}


CATTR_HOST void inline test_host_globally_parallel_locally_sequential(const int picture_count)

{
	std::cout
		<< "#################################################################################" << std::endl
		<< "Start Host tests 'GPLS'" << std::endl
		<< "#################################################################################" << std::endl;

	for each (auto task_count in TASK_COUNTS)
	{
		auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
		{
			execute_fractal_serial_each_picture(
				picture_count, SIZE, MAX_ITERATIONS, DIR_CPU_TEST);
		});

		auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
		{
			execute_fractal_parallel_each_picture(
				task_count, picture_count, SIZE, MAX_ITERATIONS, DIR_CPU_TEST);
		});

		save_and_display_host_results("HOST-GPLS", task_count, duration_thread_single, duration_thread_multiple);
	}

	std::cout
		<< "#################################################################################" << std::endl
		<< "Ended Host tests 'GPLS'" << std::endl
		<< "#################################################################################" << std::endl;
}

CATTR_HOST void inline test_host_globally_sequential_locally_parallel(const int picture_count)

{
	std::cout
		<< "#################################################################################" << std::endl
		<< "Start Host tests 'GSLP'" << std::endl
		<< "#################################################################################" << std::endl;
	
	for each (auto task_count in TASK_COUNTS)
	{

		auto duration_thread_single = mpv_runtime::run_with_measure(1, [&]
		{
			execute_fractal_serial_each_rows(
				picture_count, task_count, SIZE, MAX_ITERATIONS, DIR_CPU_TEST);
		});

		auto duration_thread_multiple = mpv_runtime::run_with_measure(1, [&]
		{
			execute_fractal_parallel_each_rows(
				task_count, picture_count, SIZE, MAX_ITERATIONS, DIR_CPU_TEST);
		});

		save_and_display_host_results("HOST-GSLP", task_count, duration_thread_multiple, duration_thread_single);
	}
	std::cout
		<< "#################################################" << std::endl
		<< "Ended Host tests 'GSLP'" << std::endl
		<< "#################################################" << std::endl;
}
#endif
