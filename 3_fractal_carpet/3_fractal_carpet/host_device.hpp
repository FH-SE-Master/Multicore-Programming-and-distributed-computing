#if !defined _hpp_host_device_
#define      _hpp_host_device_
#include <device_launch_parameters.h>
#include <array>
#include "pfc_bitmap.h"
#include "pfc_complex.h"

__device__ __device__ const std::array<pfc::RGB_4_t, 16> RGB_MAPPING{
	  pfc::RGB_4_t{ 15, 30, 66, 0 }
	, pfc::RGB_4_t{ 26, 7, 25, 0 }
	, pfc::RGB_4_t{ 47, 1, 9, 0 }
	, pfc::RGB_4_t{ 73, 4, 4, 0 }
	, pfc::RGB_4_t{ 100, 7, 0, 0 }
	, pfc::RGB_4_t{ 138, 44, 12, 0 }
	, pfc::RGB_4_t{ 177, 82, 24, 0 }
	, pfc::RGB_4_t{ 209, 125, 57, 0 }
	, pfc::RGB_4_t{ 229, 181, 134, 0 }
	, pfc::RGB_4_t{ 248, 236, 211, 0 }
	, pfc::RGB_4_t{ 191, 233, 241, 0 }
	, pfc::RGB_4_t{ 95, 201, 248, 0 }
	 ,pfc::RGB_4_t{ 0, 170, 255, 0 }
	, pfc::RGB_4_t{ 0, 128, 204, 0 }
	 ,pfc::RGB_4_t{ 0, 87, 153, 0 }
	, pfc::RGB_4_t{ 3, 52, 106, 0 } };

// https://github.com/boostorg/compute/blob/master/example/mandelbrot.cpp
// http://jonisalonen.com/2013/lets-draw-the-mandelbrot-set/

__host__ __device__ inline void calculate_fractal(const int height,
                              const int width,
                              const int max_iterations,
							  const int start_row, 
							  const int end_row, 
							  const pfc::complex<float> initial_value,
                              pfc::bitmap::pixel_t* pixels)
{
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				double c_re = (col - width / 2.0)*4.0 / width;
				double c_im = (row - height / 2.0)*4.0 / width;

				pfc::complex<float> c(c_re, c_im);
				pfc::complex<float> z(0, 0);

				int i;
				for (i = 0; i < max_iterations && norm(z) < 2.0; i++) {
					z = z * z + c;
				}

				int const color_index = i % 16;
				pixels[row*height + col] = RGB_MAPPING[color_index];
			}
		}
}
#endif
