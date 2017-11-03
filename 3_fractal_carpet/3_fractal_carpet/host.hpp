#if !defined __HOST__
#define      __HOST__

#include <stdio.h>
#include <ostream>
#include <iostream>
#include "host_device.hpp"
#include "pfc_parallel.h"

__host__ void execute_fractal_serial(const int size, const int max_iterations) {
	pfc::task_group task_group;
	pfc::bitmap bitmap{ size, size };

	std::cout << "Calculating single threaded" << std::endl << std::endl;

	calculate_fractal(size, size, max_iterations, 0, 0, pfc::complex<float>(0, 0), bitmap.get_pixels());

	bitmap.to_file("fractal-single_thread.jpg");
}

// What is begin end
__host__ void calculate_fractal_host(int chunk, int begin, int end) {
	pfc::bitmap bitmap{ 1000,1000 };
	calculate_fractal(1000, 1000, 1000, 0, 0, pfc::complex<float>(begin, end), bitmap.get_pixels());
}

__host__ void execute_fractal_parallel(const int count, const int size, const int max_iterations) {
	pfc::task_group task_group{};
	std::cout << "Calculating " + std::to_string(count) + " fractal images" << std::endl;

	pfc::parallel_range(task_group, 40, count, calculate_fractal_host);
}

#endif