#include <iomanip>
#include <iostream>
#include <time.h>
#include <map>

/*
 *		MANUAL PROFILING FOR CUSTOM SCOPE
 */

#ifdef ENABLE_PROFILING
#define START_PROFILING ExtProfiler::Instance().StartTimer();
#define STOP_PROFILING(name) ExtProfiler::Instance().StopTimer(name);
#define PRINT_PROFILING_RESULTS ExtProfiler::Instance().PrintResults();
#else
#define START_PROFILING
#define STOP_PROFILING(name)
#define PRINT_PROFILING_RESULTS
#endif

class ExtProfiler
{
#define MS 1000.0
#define NS 1000000000.0
#define NS_MS 1000000.0

	typedef struct 
	{
		double time;
		int calls;
	} profile;

	struct timespec start;
	struct timespec end;
	std::map<std::string, profile> profiles;

	ExtProfiler(){};
	ExtProfiler(ExtProfiler const&);
	void operator=(ExtProfiler const&);

	public:
	static ExtProfiler& Instance()
	{
		static ExtProfiler instance;
		return instance;
	}

	void StartTimer()
	{
		clock_gettime(CLOCK_MONOTONIC, &start);
	}

	void StopTimer(std::string name)
	{
		clock_gettime(CLOCK_MONOTONIC, &end);

		if(profiles.find(name) == profiles.end())
		{
			profiles[name].time = 0.0;
			profiles[name].calls = 0;
		}

		long elapsedSec = end.tv_sec-start.tv_sec;
		long elapsedNS = end.tv_nsec-start.tv_nsec;

		double elapsed = (elapsedNS < 0) 
			? (elapsedSec-1)*MS + (NS+elapsedNS)/NS_MS 
			: elapsedSec*MS + elapsedNS/NS_MS;

		profiles[name].time += elapsed;
		profiles[name].calls++;
	}

	// Prints out in first call order.
	void PrintResults()
	{
		std::cout << std::setiosflags(std::ios::fixed) << std::left << std::endl
			<< std::setw(30) << "Kernel Name" << std::setw(15) << "Time (ms)"
			<< std::setw(10) << "Calls" << std::setw(10) << "ms/call"
			<< std::setw(10) << "Efficiency" << std::endl;

		for(auto profile : profiles)
		{
			std::cout << std::setw(30) << profile.first
				<< std::setw(15) << std::setprecision(3) << profile.second.time
				<< std::setw(10) << profile.second.calls
				<< std::setw(10) << profile.second.time/profile.second.calls
				<< std::setw(10) << 1/(profile.second.time/profile.second.calls) 
				<< std::endl;
		}

		std::cout << std::endl;
	}
};

