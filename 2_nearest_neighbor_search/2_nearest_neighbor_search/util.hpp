#if !defined _h_util_
#define      _h_util_
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <random>
#include <iostream>
#include "../../1_hello_world/HelloWorld/HelloWorld/pfc_cuda_device_info.h"

namespace mpv_exception
{
	// Custom cuda exception which provides all cuda related error and information about error line in the source file.
	class cuda_exception: std::runtime_error
	{

	public:
		cuda_exception(cudaError const error, std::string const & file, int const line) : std::runtime_error(create_message(error, file, line)) {
		}

	private:
		static std::string create_message(cudaError const error, std::string const & file, int const line) {
			std::string message = "CUDA error: ";

			message += std::to_string(error) + " '" + cudaGetErrorString(error) + "' occured. \n";

			if (!file.empty() && (line > 0)) {
				message += "File '" + file + "' line: '" + std::to_string(line) + "'";
			}

			return std::move(message);
		}
	};

	// Check for error, used when a file is provided
	// throws cuda_exception which contains error information
	inline void check(cudaError const error, std::string const & file, int const line) {
		if (error != cudaSuccess) {
			throw cuda_exception(error, file, line);
		}
	}
	
	// Checks for error, used when no file is provided
	// @see check(cudaError const, std::string const&, int const)
	inline void check(cudaError const error) {
		check(error, "", 0);
	}
}

namespace mpv_chrono
{
	class timer final
	{

	private:
		using timer_type = std::chrono::high_resolution_clock;
		
		timer_type::time_point start_{};
		timer_type::time_point end_{};

	public:
		void start()
		{
			start_ = timer_type::now();
		}
		
		auto get_duration()
		{
			end_ = timer_type::now();
			return end_ - start_;
		}

		auto stop()
		{
			end_ = timer_type::now();
			return get_duration();
		}

		void reset()
		{
			start_ = end_ = timer_type::time_point{};
		}
	};
}

namespace mpv_random
{
	template <typename T> inline T get_random_uniformed(T const l, T const u) {
		if(l <= 0 || u <= 0 || l>= u)
		{
			throw std::runtime_error("Lower and upper bound are invalid. Lower: " + std::to_string(l) + " / Upper: " + std::to_string(u));
		}
		std::random_device rd; 
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(l, u);

		return dis(gen);
	}
}

namespace mpv_device
{
	inline void print_device_info()
	{
		auto const device_info = pfc::cuda::get_device_info();
		auto const device_props = pfc::cuda::get_device_props();

		std::cout << "compute capability: " << device_info.cc_major << "." << device_info.cc_minor << std::endl;
		std::cout << "Device: " << device_props.name << std::endl;
	}
}

#endif
