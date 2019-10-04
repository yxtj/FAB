#include "Sleeper.h"
#include <chrono>
#include <thread>

using namespace std;

long long Sleeper::_sleep_overhead_ns = 51000; // at least 50 micro-seconds by Linux Kernel Implementation
long long Sleeper::_measure_overhead_ns = 80;

using hrc = chrono::high_resolution_clock;

// static

void Sleeper::Init()
{
	long long s = 0;
	for(int i = 0; i < 10; ++i){
		hrc::time_point f = hrc::now();
		this_thread::sleep_for(chrono::nanoseconds(10000)); // sleep less than 1000 ns is omitted by some compiler and OS
		hrc::time_point l = hrc::now();
		hrc::duration d = l - f;
		s += chrono::duration_cast<chrono::nanoseconds>(d).count();
	}
	_sleep_overhead_ns = s / 10;
	s = 0;
	for(int i = 0; i < 10; ++i){
		hrc::time_point f = hrc::now();
		hrc::time_point l = hrc::now();
		hrc::duration d = l - f;
		s += chrono::duration_cast<chrono::nanoseconds>(d).count();
	}
	_measure_overhead_ns = s / 10;
}

long long Sleeper::GetSleepOverhead()
{
	return _sleep_overhead_ns;
}

long long Sleeper::GetMeasureOverhead()
{
	return _measure_overhead_ns;
}

void Sleeper::SleepYield(const double seconds)
{
	this_thread::sleep_for(chrono::duration<double>(seconds));
}

void Sleeper::SleepYield(const long long nano_seconds)
{
	this_thread::sleep_for(chrono::nanoseconds(nano_seconds));
}

void Sleeper::SleepLoop(const double seconds)
{
	SleepLoop(static_cast<long long>(seconds * 1e9));
}

void Sleeper::SleepLoop(const long long nano_seconds)
{
	long long nsEnd = hrc::now().time_since_epoch().count() + static_cast<long long>(nano_seconds);
	long long now;
	do{
		now = hrc::now().time_since_epoch().count();
	} while(now < nsEnd);
}

void Sleeper::Sleep(const double seconds)
{
	Sleep(static_cast<long long>(seconds * 1e9));
}

void Sleeper::Sleep(const long long nano_seconds)
{
	if(nano_seconds <= 0){
		return;
	} else if(nano_seconds >= _sleep_overhead_ns){
		SleepYield(nano_seconds);
	} else{
		SleepLoop(nano_seconds);
	}
}

// non-static

void Sleeper::sleep(const double seconds)
{
	sleep(static_cast<long long>(seconds * 1e9));
}

void Sleeper::sleep(const long long nano_seconds)
{
	long long f = chrono::duration_cast<chrono::nanoseconds>(hrc::now().time_since_epoch()).count();
	Sleep(nano_seconds - credit - _measure_overhead_ns - _measure_overhead_ns);
	long long l = chrono::duration_cast<chrono::nanoseconds>(hrc::now().time_since_epoch()).count();
	long long c = l - f;
	credit += c - nano_seconds;
}

long long Sleeper::getCredit() const
{
	return credit;
}