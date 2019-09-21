/*
* SyncUnit.cpp
*
*  Created on: Jan 19, 2017
*      Author: tzhou
*/
#include "SyncUnit.h"

void SyncUnit::wait() {
	if(flag)	return;
	std::unique_lock<std::mutex> ul(m);
	if(flag)	return;
	cv.wait(ul, [&]() {return flag; });
}

bool SyncUnit::wait_for(const double dur) {
	std::unique_lock<std::mutex> ul(m);
	return cv.wait_for(ul, std::chrono::duration<double>(dur), [&]() {return flag; });
}

void SyncUnit::wait_n_reset() {
	std::unique_lock<std::mutex> ul(m);
	cv.wait(ul, [&]() {return flag; }); // directly return if flag==true
	flag = false;
}

bool SyncUnit::wait_n_reset_for(const double dur) {
	std::unique_lock<std::mutex> ul(m);
	bool ret = cv.wait_for(ul, std::chrono::duration<double>(dur), [&]() {return flag; });
	flag = false;
	return ret;
}

void SyncUnit::notify() {
	{
		std::unique_lock<std::mutex> ul(m);
		flag = true;
	}
	cv.notify_all();
}

void SyncUnit::reset() {
	std::unique_lock<std::mutex> ul(m);
	flag = false;
}
