#pragma once
#include <chrono>

class Timer
{
	using clock = std::chrono::high_resolution_clock;
	clock::time_point _time;

	static clock::time_point _boot_time;
	static long long _sleep_unit_ns;
	static double _sleep_unit_s;
public:
	Timer();
	void restart();

	long long elapseMS() const;
	long long elapseUS() const;
	long long elapseNS() const;
	int elapseS() const;
	double elapseSd() const;
	double elapseMin() const;

	static void Init();
	static double Now();
	static double NowSinceBoot();

	static void Sleep(const double seconds, const bool exact = false);
	static void Sleep(const long long nano_seconds, const bool exact = false);
};
