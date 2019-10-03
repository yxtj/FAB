#include "Timer.h"
#include <thread>

using namespace std;

std::chrono::high_resolution_clock::time_point Timer::_boot_time = clock::now();
long long Timer::_sleep_unit_ns = 51000; // at least 50 micro-seconds by Linux Kernel Implementation
double Timer::_sleep_unit_s = 5.1e-5;

Timer::Timer()
{
	restart();
}

void Timer::restart()
{
	_time = clock::now();
}

long long Timer::elapseMS() const
{
	return chrono::duration_cast<chrono::milliseconds>(
		clock::now() - _time).count();
}

long long Timer::elapseUS() const
{
	return chrono::duration_cast<chrono::microseconds>(
		clock::now() - _time).count();
}

long long Timer::elapseNS() const
{
	return chrono::duration_cast<chrono::nanoseconds>(
		clock::now() - _time).count();
}

int Timer::elapseS() const
{
	return chrono::duration_cast<chrono::duration<int>>(
		clock::now() - _time).count();
}

double Timer::elapseSd() const
{
	return chrono::duration_cast<chrono::duration<double>>(
		clock::now() - _time).count();
}

double Timer::elapseMin() const
{
	chrono::duration<double, ratio<60> > passed = clock::now() - _time;
	return passed.count();
}

// static

void Timer::Init()
{
	_boot_time = clock::now();
	long long s = 0;
	for(int i = 0; i < 10; ++i){
		clock::time_point f = clock::now();
		this_thread::sleep_for(chrono::nanoseconds(1));
		clock::time_point l = clock::now();
		s += (l - f).count();
	}
	_sleep_unit_ns = s / 10;
	_sleep_unit_s = static_cast<double>(s) / 10;
}

double Timer::Now()
{
	return chrono::duration<double>(
		clock::now().time_since_epoch()).count();
}

double Timer::NowSinceBoot()
{
	return chrono::duration_cast<chrono::duration<double>>(
		clock::now() - _boot_time).count();
}

void Timer::Sleep(const double seconds, const bool exact)
{
	if(!exact || seconds >= _sleep_unit_s)
		this_thread::sleep_for(chrono::duration<double>(seconds));
	long long nsEnd = clock::now().time_since_epoch().count() + static_cast<long long>(seconds * 1e9);
	long long now;
	do{
		now = clock::now().time_since_epoch().count();
	} while(now < nsEnd);
}

void Timer::Sleep(const long long nano_seconds, const bool exact)
{
	if(!exact || nano_seconds >= _sleep_unit_ns)
		this_thread::sleep_for(chrono::nanoseconds(nano_seconds));
	long long nsEnd = clock::now().time_since_epoch().count() + static_cast<long long>(nano_seconds);
	long long now;
	do{
		now = clock::now().time_since_epoch().count();
	} while(now < nsEnd);
}
