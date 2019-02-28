#pragma once
#include <vector>
#include <string>

/**
Get an interval between two consecutive synchronization
*/
struct IntervalEstimator {
	// param : the method-name and parameters used to generate interval
	// nWorker, nPoint : total number of workers and data points
	virtual void init(const std::vector<std::string>& param,
		const size_t nWorker, const size_t nPoint);

	// get the time to wait
	virtual double interval() = 0;
	// Core function: update the internal state
	// improve, interval, points: the improvement, time interval, # of data point used since last synchronization
	// timeSync, timeSince: the time used for a synchrnoization, time since the training start
	virtual void update(const double improve, const double interval, const size_t points,
		const double timeSync, const double timeStart) = 0;
	// by default, it uses averged l2-norm to merge improveVec into one value and call the previous one
	virtual void update(const std::vector<double>& improveVec, const double interval, const size_t points,
		const double timeSync, const double timeStart);

	virtual ~IntervalEstimator(){};

protected: // helper functions
	void mu_forward_vec_ignore(const std::vector<double>& improve, const double interval, const size_t points,
		const double timeSync, const double timeStart);
	void mu_forward_vec_l1(const std::vector<double>& improve, const double interval, const size_t points,
		const double timeSync, const double timeStart);
	void mu_forward_vec_l2(const std::vector<double>& improve, const double interval, const size_t points,
		const double timeSync, const double timeStart);

protected: // state
	size_t nw; // number of workers
	size_t np; // number of points
};

struct IntervalEstimatorFactory{
	// param[0] is the name of strategy
	// Support: "interval" -> fixed time interval (param: second)
	//          "portion" -> fixed portion of data points (param: portion)
	//          "improve" -> fixed improvement amount (param: avg-improve, max-wait-time)
	//          "balance" -> maximize the progress by balancing computing time and synchronization time (param: num-of-window)
	static IntervalEstimator* generate(const std::vector<std::string>& param,
		const size_t nWorker, const size_t nPoint);
};
