#include "Timer.h"
#include <thread>
#include <iostream>

using namespace std;

Timer::clock::time_point Timer::_boot_time = clock::now();

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
