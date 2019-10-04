#pragma once

class Sleeper{
	static long long _sleep_overhead_ns;
	static long long _measure_overhead_ns;

	long long credit = 0; // how many more nano-seconds spent in sleeping
public:
	// probe the sleep overhead
	static void Init();
	static long long GetSleepOverhead();
	static long long GetMeasureOverhead();

	// calling sleep API which yields the CPU
	static void SleepYield(const double seconds);
	static void SleepYield(const long long nano_seconds);
	// keep checking the time
	static void SleepLoop(const double seconds);
	static void SleepLoop(const long long nano_seconds);

	// try to be more accurate
	static void Sleep(const double seconds);
	static void Sleep(const long long nano_seconds);

	// sleep less if it sleeps longer last time
	void sleep(const double seconds);
	void sleep(const long long nano_seconds);

	long long getCredit() const;
};
