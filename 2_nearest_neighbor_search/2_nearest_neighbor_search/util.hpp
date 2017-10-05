#if !defined _h_exception_
#define      _h_exception_
#include <chrono>

namespace mpv_exception
{
	inline void check_for_error()
	{
	}
}

namespace mpv_chrono
{
	class timer final
	{
	private:
		using timer_type = std::chrono::high_resolution_clock;
		using duration_time = std::chrono::duration<float, std::ratio<1, 1000000>>;
		using time_point_type = std::chrono::time_point<std::chrono::high_resolution_clock>;
		
		timer_type timer_{};
		time_point_type start_{};
		time_point_type end_{};

	public:
		using timer_type = std::chrono::high_resolution_clock;
		using duration_time = std::chrono::duration<float, std::ratio<1, 1000000>>;
		using time_point_type = std::chrono::time_point<std::chrono::high_resolution_clock>;

		void start()
		{
			start_ = timer_type::now();
		}

		duration_time stop()
		{
			end_ = timer_type::now();
			return get_duration();
		}

		void reset()
		{
			start_ = time_point_type{};
			end_ = time_point_type{};;
		}

		duration_time get_duration()
		{
			end_ = timer_type::now();
			return start_ - end_;
		}
	};
}

namespace mpv_random
{
}


#endif
