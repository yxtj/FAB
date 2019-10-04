#include "EM.h"
#include "util/Timer.h"
#include "util/Sleeper.h"
#include <thread>
#include <stdexcept>

using namespace std;

void EM::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		if(rate < 0)
			rate = -rate;
	} catch(...){
		throw invalid_argument("Cannot parse parameters for EM");
	}
}

std::string EM::name() const
{
	return "em";
}

void EM::setRate(const double rate) {
	if(rate >= 0)
		this->rate = rate;
	else
		this->rate = -rate;
}

double EM::getRate() const {
	return rate;
}

void EM::prepare()
{
	// initialize hidden variable
	int nh = pm->getKernel()->lengthHidden();
	h.assign(pd->size(), vector<double>(nh));
	// initialize parameter by data
	if(pm->getKernel()->needInitParameterByData()){
		Parameter p;
		p.init(pm->paramWidth(), 0.0);
		pm->setParameter(p);
		size_t s = pd->size();
		for(size_t i = 0; i < s; ++i){
			pm->getKernel()->initVariables(
				pd->get(i).x, pm->getParameter().weights, pd->get(i).y, nullptr);
		}
	}
}

Trainer::DeltaResult EM::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		auto g = pm->gradient(pd->get(i), &h[i]);
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
	}
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}
	return { i - start, i - start, move(grad) };
}

Trainer::DeltaResult EM::batchDelta(std::atomic<bool>& cond,
	const size_t start, const size_t cnt, const bool avg, const double adjust)
{
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	double loss = 0.0;
	vector<double> grad(nx, 0.0);
	Sleeper slp;
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		Timer tt;
		//auto p = pm->forward(pd->get(i));
		//loss += pm->loss(p, pd->get(i).y);
		//auto g = pm->backward(pd->get(i), &h[i]);
		auto g = pm->gradient(pd->get(i), &h[i]);
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
		double time = tt.elapseSd();
		slp.sleep(time * adjust);
	}
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}
	return { i - start, i - start, move(grad), loss };
}
