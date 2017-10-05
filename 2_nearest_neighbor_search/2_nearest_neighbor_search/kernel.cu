#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.hpp"

#include <stdio.h>
#include <thread>
#include <iostream>

__global__ void kernel()
{
}

int main()
{
	mpv_chrono::timer timer{};
	timer.start();
	for (auto i = 0; i < 10; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		timer.stop();
		std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(timer.get_duration()).count() <<
			std::endl;
	}
}
