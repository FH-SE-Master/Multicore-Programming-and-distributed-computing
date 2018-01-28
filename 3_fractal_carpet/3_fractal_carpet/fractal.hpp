#if !defined _hpp_fractal_
#define      _hpp_fractal_
#include "pfc_cuda_device_info.h"
#include "constant.h"
#include "util.hpp"
#include <device_functions.h>

// https://github.com/boostorg/compute/blob/master/example/mandelbrot.cpp
// http://jonisalonen.com/2013/lets-draw-the-mandelbrot-set/

CATTR_DEVICE void calculate_fractal_part(const int size,
	const int max_iterations,
	const int row,
	const int col,
	const pfc::complex<float> initial_value,
	pfc::bitmap::pixel_t* pixels,
	pfc::bitmap::pixel_t* rgb_map) {

	#pragma unroll_completely
	if (row < size && col < size) {
		auto c_re = __fmul_rn(__fdiv_rn((col - size), 2.0f), 4.0f) / size;
<<<<<<< HEAD
		auto c_im = __fmul_rn(__fdiv_rn((row - size), 2.0f),  4.0f) / size;
=======
		auto c_im = __fmul_rn(__fdiv_rn((row - size), 2.0f), 4.0f) / size;
>>>>>>> Branch_double_to_float

		pfc::complex<float> c(c_re, c_im);
		pfc::complex<float> z(0.0f, 0.0f);

		int i;
		for (i = 0; i < max_iterations && norm(z) < 2.0f; i++)
		{
			z = z * z + c;
		}

		pixels[row * size + col] = rgb_map[(i % RGB_COLOR_SIZE)];
	}
}

CATTR_DEVICE void calculate_fractal(const int size,
	const int max_iterations,
	const int start_row,
	const int end_row,
	const pfc::complex<float> initial_value,
	pfc::bitmap::pixel_t* pixels,
	pfc::bitmap::pixel_t* rgb_map)
{
	#pragma unroll_completely
	for (auto row = start_row; row < end_row; row++)
	{
		for (auto col = 0; col < size; col++)
		{
			calculate_fractal_part(size, max_iterations, row, col, initial_value, pixels, rgb_map);
		}
	}
}
#endif
