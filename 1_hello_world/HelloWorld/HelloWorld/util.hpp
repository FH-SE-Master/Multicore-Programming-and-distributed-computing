#if !defined _h_util_
#define      _h_util_

#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <random>
#include <iostream>
#include "./pfc_cuda_device_info.h"
#include <future>
#include <thread>

using namespace std::literals;

#undef  CUDA_FREE
#define CUDA_FREE(dp_mem) \
	mpv_memory::free (dp_mem, __FILE__, __LINE__)

#undef  CUDA_MALLOC
#define CUDA_MALLOC(T, size) \
	mpv_memory::malloc <T> (size, __FILE__, __LINE__)

#undef  CUDA_MEMCPY
#define CUDA_MEMCPY(p_dst, p_src, size, kind) \
   mpv_memory::memcpy(p_dst, p_src, size, kind, __FILE__, __LINE__)

namespace mpv_exception
{
	// Custom cuda exception which provides all cuda related error and information about error line in the source file.
	class cuda_exception : std::runtime_error
	{
	public:
		cuda_exception(cudaError const error, std::string const& file, int const line) : std::runtime_error(
			create_message(error, file, line))
		{
		}

	private:
		static std::string create_message(cudaError const error, std::string const& file, int const line)
		{
			std::string message = "CUDA error: ";

			message += std::to_string(error) + " '" + cudaGetErrorString(error) + "' occured. \n";

			if (!file.empty() && (line > 0))
			{
				message += "File '" + file + "' line: '" + std::to_string(line) + "'";
			}

			return std::move(message);
		}
	};

	// Check for error, used when a file is provided
	// throws cuda_exception which contains error information
	inline void check(cudaError const error, std::string const& file, int const line)
	{
		if (error != cudaSuccess)
		{
			throw cuda_exception(error, file, line);
		}
	}

	// Checks for error, used when no file is provided
	// @see check(cudaError const, std::string const&, int const)
	inline void check(cudaError const error)
	{
		check(error, "", 0);
	}
} // mpv_exception

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
} // mpv_chrono

namespace mpv_random
{
	template <typename T>
	inline T get_random_uniformed(T const l, T const u)
	{
		if (l <= 0 || u <= 0 || l >= u)
		{
			throw std::runtime_error(
				"Lower and upper bound are invalid. Lower: " + std::to_string(l) + " / Upper: " + std::to_string(u));
		}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(l, u);

		return dis(gen);
	} // get_random_uniformed
} // mpv_random

namespace mpv_device
{
	inline void print_device_info()
	{
		auto const device_info = pfc::cuda::get_device_info();
		auto const device_props = pfc::cuda::get_device_props();

		std::cout << "compute capability: " << device_info.cc_major << "." << device_info.cc_minor << std::endl;
		std::cout << "Device: " << device_props.name << std::endl;
	} // print_device_info
} // mpv_device

namespace mpv_threading
{
	template <typename fn_type, typename ...arg_type>
	void execute_parallel(int const n, fn_type&& fun, arg_type&& ...args)
	{
		std::vector<std::future<void>> task_group;

		task_group.push_back(std::async(std::launch::async, [&]
	                                {
		                                std::invoke(std::forward<fn_type>(fun), std::forward<arg_type>(args)...);
	                                }));

		for (auto& f : task_group)
		{
			f.get();
		}
	} // execute_parallel

	inline void warm_up_cpu(std::chrono::seconds const how_long = 5s)
	{
		auto const cores = std::max<int>(1, std::thread::hardware_concurrency());
		auto const start = std::chrono::high_resolution_clock::now();

		std::vector<std::thread> group;

		execute_parallel(cores, [&]
	                 {
		                 // busy waiting for how_long seconds
		                 while (std::chrono::duration_cast<std::chrono::seconds>
			                 (std::chrono::high_resolution_clock::now() - start) < how_long);
	                 });
	} // warm_up_cpu
} // mpv_threading

namespace mpv_runtime
{
	template <typename fn_type, typename ...arg_type>
	auto run_with_measure(int const n, fn_type&& fun, arg_type&& ...args)
	{
		mpv_chrono::timer timer{};

		if (n > 0)
		{
			timer.start();
			for (int i{0}; i < n; ++i)
			{
				std::invoke(std::forward<fn_type>(fun), std::forward<arg_type>(args)...);
			}
			timer.stop();
		}

		return timer.get_duration();
	} // run_with_measure

	template <typename duration_t>
	inline double speedup(duration_t const& first, duration_t const& second)
	{
		static typename duration_t::rep const zero = 0;
		return (second.count() != zero) ? 1.0 * first.count() / second.count() : 0;
	} // speedup
} // mpv_runtime

namespace mpv_memory
{
	///allocates the given elements on the device
	template <typename T>
	T* malloc(size_t size, std::string const& file = "", int const line = 0)
	{
		T* dp_mem = nullptr;

		if (size > 0)
		{
			mpv_exception::check(cudaMalloc(&dp_mem, size * sizeof(T)), file, line);
		}
		return dp_mem;
	} // malloc

	///frees the given pointer if not nullptr
	template <typename T>
	inline T* & free(T* & dp_mem, std::string const& file = "", int const line = 0)
	{
		if (dp_mem != nullptr)
		{
			mpv_exception::check(cudaFree(dp_mem), file, line);
			dp_mem = nullptr;
		}
		return dp_mem;
	} // free

	///wraps a memcpy call 
	template <typename T>
	inline T* memcpy(T* const p_dst, T const* const p_src, size_t size, cudaMemcpyKind const kind,
	                 std::string const& file = "", int const line = 0)
	{
		if ((p_dst != nullptr) && (p_src != nullptr) && (size > 0))
		{
			mpv_exception::check(cudaMemcpy(p_dst, p_src, size * sizeof(T), kind), file, line);
		}

		return p_dst;
	} // memcpy
} // mpv_memory
#endif
