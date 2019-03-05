/*
* SyncUnit.cpp
*
*  Created on: Jan 19, 2017
*      Author: tzhou
*/
#include "SyncUnit.h"

void SyncUnit::wait() {
	if(ready)	return;
	std::unique_lock<std::mutex> ul(m);
	if(ready)	return;
	cv.wait(ul, [&]() {return ready; });
}

bool SyncUnit::wait_for(const double dur) {
	std::unique_lock<std::mutex> ul(m);
	return cv.wait_for(ul, std::chrono::duration<double>(dur), [&]() {return ready; });
}

void SyncUnit::wait_n_reset() {
	std::unique_lock<std::mutex> ul(m);
	cv.wait(ul, [&]() {return ready; }); // directly return if ready==true
	ready = false;
}

bool SyncUnit::wait_n_reset_for(const double dur) {
	std::unique_lock<std::mutex> ul(m);
	bool ret = cv.wait_for(ul, std::chrono::duration<double>(dur), [&]() {return ready; });
	ready = false;
	return ret;
}

void SyncUnit::notify() {
	{
		std::unique_lock<std::mutex> ul(m);
		ready = true;
	}
	cv.notify_all();
}

void SyncUnit::reset() {
	std::unique_lock<std::mutex> ul(m);
	ready = false;
}
