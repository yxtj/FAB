#include "EM.h"
#include "util/Timer.h"
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
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;
	for(i = start; i < end; ++i){
		auto g = pm->gradient(pd->get(i), &h[i]);
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
	}
	if(start != end){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (end - start);
		for(auto& v : grad)
			v *= factor;
	}
	return { i - start, i - start, move(grad) };
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
	vector<double> grad(nx, 0.0);
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		Timer tt;
		auto g = pm->gradient(pd->get(i), &h[i]);
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
		double time = tt.elapseSd();
		this_thread::sleep_for(chrono::duration<double>(time));
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
