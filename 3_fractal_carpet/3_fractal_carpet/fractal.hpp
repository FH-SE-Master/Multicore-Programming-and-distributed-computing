#if !defined _hpp_fractal_
#define      _hpp_fractal_
#include "pfc_cuda_device_info.h"
#include "constant.h"
#include "util.hpp"

// https://github.com/boostorg/compute/blob/master/example/mandelbrot.cpp
// http://jonisalonen.com/2013/lets-draw-the-mandelbrot-set/

CATTR_HOST_DEV_INL void calculate_fractal_part(const int size,
	const int max_iterations,
	const int row,
	const int col,
	const pfc::complex<float> initial_value,
	pfc::bitmap::pixel_t* pixels,
	pfc::bitmap::pixel_t* rgb_map) {

	if (row < size && col < size) {
		auto c_re = (col - size / 2.0) * 4.0 / size;
		auto c_im = (row - size / 2.0) * 4.0 / size;

		pfc::complex<float> c(c_re, c_im);
		pfc::complex<float> z(0, 0);

		int i;
		for (i = 0; i < max_iterations && norm(z) < 2.0; i++)
		{
			z = z * z + c;
		}

		pixels[row * size + col] = rgb_map[(i % RGB_COLOR_SIZE)];
	}
}

CATTR_HOST_DEV_INL void calculate_fractal(const int size,
	const int max_iterations,
	const int start_row,
	const int end_row,
	const pfc::complex<float> initial_value,
	pfc::bitmap::pixel_t* pixels,
	pfc::bitmap::pixel_t* rgb_map)
{
	for (auto row = start_row; row < end_row; row++)
	{
		for (auto col = 0; col < size; col++)
		{
			calculate_fractal_part(size, max_iterations, row, col, initial_value, pixels, rgb_map);
		}
	}
}
#endif
