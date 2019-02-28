#include "IntervalEstimator.h"
#include "math/norm.h"
#include <utility>
#include <algorithm>
#include <cmath>

using namespace std;

/*template <typename T>
static T max(const T& a, const T& b){
	return a < b ? b : a;
}
template <typename T>
static T min(const T& a, const T& b){
	return a < b ? a : b;
}
*/
// ---- base interval estimator ----

void IntervalEstimator::init(const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint)
{
	nw = nWorker;
	np = nPoint;
}

void IntervalEstimator::update(const std::vector<double>& improveVec, const double interval, const size_t points,
	const double timeSync, const double timeStart)
{
	mu_forward_vec_l2(improveVec, interval, points, timeSync, timeStart);
}

void IntervalEstimator::mu_forward_vec_ignore(const std::vector<double>& improve,
	const double interval, const size_t points,
	const double timeSync, const double timeStart)
{
	return this->update(0.0, interval, points, timeSync, timeStart);
}
void IntervalEstimator::mu_forward_vec_l1(const std::vector<double>& improve,
	const double interval, const size_t points,
	const double timeSync, const double timeStart)
{
	return this->update(l1norm(improve), interval, points, timeSync, timeStart);
}
void IntervalEstimator::mu_forward_vec_l2(const std::vector<double>& improve,
	const double interval, const size_t points,
	const double timeSync, const double timeStart)
{
	return this->update(l2norm(improve, true), interval, points, timeSync, timeStart);
}

// ---- fixed time ----

struct IntervalEstimatorFixed : public IntervalEstimator{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint){
		IntervalEstimator::init(param, nWorker, nPoint);
		fixedInterval = stod(param[1]);
	}
	virtual void update(const double improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){}
	virtual void update(const std::vector<double>& improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){}
	virtual double interval(){
		return fixedInterval;
	}
private:
	double fixedInterval;
};

// ---- fixed portion ----

struct IntervalEstimatorPortion : public IntervalEstimator{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint){
		IntervalEstimator::init(param, nWorker, nPoint);
		fixedPortion = stod(param[1]);
		size_t t = static_cast<size_t>(ceil(nPoint*fixedPortion));
		fixedPoint = max(t, (size_t)1);
		estRatio = 100;
		state = 0.01;
	}
	virtual void update(const double improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){
		estRatio = (estRatio + points / interval) / 2;
		state = (state + estRatio*fixedPoint) / 2;
	}
	virtual void update(const std::vector<double>& improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){
		mu_forward_vec_ignore(improve, interval, points, timeSync, timeStart);
	}
	virtual double interval(){
		return state;
	}
private:
	double fixedPortion;
	size_t fixedPoint;
	double estRatio;
	double state;
};

// ---- fixed improvement ----

struct IntervalEstimatorImprove : public IntervalEstimator{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint){
		IntervalEstimator::init(param, nWorker, nPoint);
		fixedImprove = stod(param[1]);
		maxWaiting = param.size() > 2 ? stod(param[2]) : 1.0;
		
		estRatio = 0.01;
		state = 0.01;
	}
	virtual void update(const double improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){
		estRatio = (estRatio + improve / interval) / 2;
		state = (state + estRatio * fixedImprove) / 2;
		state = min(state, maxWaiting);
	}
	virtual double interval(){
		return state;
	}
private:
	double fixedImprove;
	double maxWaiting;

	double estRatio;
	double state;
};

// ---- fixed balance ----

struct IntervalEstimatorBalance: public IntervalEstimator{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint){
		IntervalEstimator::init(param, nWorker, nPoint);
		nw = param.size() > 1 ? stoi(param[1]) : 2;
		oldBuff.assign(nw, pair<double, double>(0, 0.01));
		oldImproveSum = 0;
		oldIntervalSum = nw * 0.01;
		p = 0;
		estSyncTime = 0.01;
		state = 0.01;
	}
	virtual void update(const double improve, const double interval, const size_t points,
		const double timeSync, const double timeStart){
		double newRatio = improve / (interval+timeSync);
		double oldRatio = oldImproveSum / (oldIntervalSum + estSyncTime*nw);
		estSyncTime = (estSyncTime + timeSync) / 2;
		double t = oldIntervalSum / nw;
		if(newRatio > oldRatio){
			state += t;
		} else{
			state = min(t, state);
		}
	}
	virtual double interval(){
		return state;
	}
private:
	void updateBuff(const double improve, const double interval){
		oldImproveSum -= oldBuff[p].first;
		oldIntervalSum -= oldBuff[p].second;
		oldImproveSum += improve;
		oldIntervalSum += interval;
		oldBuff[p].first = improve;
		oldBuff[p].second += interval;
		++p;
		if(p >= nw)
			p = 0;
	}
	int nw; // number of window

	double oldImproveSum, oldIntervalSum;
	vector<pair<double, double>> oldBuff;
	int p;
	
	double estSyncTime;
	double state;
};
// ---- generate ----

IntervalEstimator* IntervalEstimatorFactory::generate(
	const std::vector<std::string>& param, const size_t nWorker, const size_t nPoint)
{
	IntervalEstimator * p = nullptr;
	const string& name = param[0];
	if(name == "interval") {
		p = new IntervalEstimatorFixed();
	} else if(name == "portion"){
		p = new IntervalEstimatorPortion();
	} else if(name == "improve"){
		p = new IntervalEstimatorImprove();
	} else if(name == "balance"){
		p = new IntervalEstimatorBalance();
	}
	if(p != nullptr)
		p->init(param, nWorker, nPoint);
	return p;
}
