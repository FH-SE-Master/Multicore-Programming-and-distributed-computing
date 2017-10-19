#if !defined _hpp_host_device_
#define      _hpp_host_device_
#include <device_launch_parameters.h>
#include "pfc_bitmap.h"

// https://github.com/boostorg/compute/blob/master/example/mandelbrot.cpp
// http://jonisalonen.com/2013/lets-draw-the-mandelbrot-set/

inline void calculate_fractal(const int height,
                              const int width,
                              const int max_iterations,
                              pfc::bitmap::pixel_t* bitmap,
                              std::pair<int, int> position)
{
	if (position.first < 0 && position.second < 0)
	{
		throw std::runtime_error("Fractal cannot be positioned on such indicies");
	}

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			double c_re = (col - (width * (position.second + 1)) / 2.0) * 4.0 / (width * (position.second + 1));
			double c_im = (row - (height * (position.first + 1)) / 2.0) * 4.0 / (height * (position.first + 1));
			double x = 0, y = 0;
			int iteration = 0;
			while (x * x + y * y <= 4 && iteration < max_iterations)
			{
				double x_new = x * x - y * y + c_re;
				y = 2 * x * y + c_im;
				x = x_new;
				iteration++;
			}
			const int idx = row * width + col;
			if (iteration < max_iterations)
			{
				bitmap[idx].blue = 255;
				bitmap[idx].green = 255;
				bitmap[idx].red = 255;
			}
			else
			{
				bitmap[idx].blue = 0;
				bitmap[idx].green = 0;
				bitmap[idx].red = 0;
			}
		}
	}
};
#endif
