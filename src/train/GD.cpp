#include "GD.h"
#include "util/Timer.h"
#include "logging/logging.h"
#include <thread> // for sleep
#include <exception>

using namespace std;

void GD::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		if(rate < 0)
			rate = -rate;
	} catch(...){
		throw invalid_argument("Cannot parse parameters for GD");
	}
}

std::string GD::name() const
{
	return "gd";
}

void GD::setRate(const double rate) {
	if(rate >= 0)
		this->rate = rate;
	else
		this->rate = -rate;
}

double GD::getRate() const {
	return rate;
}

void GD::prepare()
{
	if(!pm->getKernel()->needInitParameterByData())
		return;
	Parameter p;
	p.init(pm->paramWidth(), 0.0);
	pm->setParameter(p);
	size_t s = pd->size();
	for(size_t i = 0; i < s; ++i){
		pm->getKernel()->initVariables(
			pd->get(i).x, pm->getParameter().weights, pd->get(i).y, nullptr);
	}
}

GD::~GD()
{
	LOG(INFO) << "[Stat] time-grad-calc: " << stat_t_grad_calc
		<< " time-grad-post: " << stat_t_grad_post;
}

Trainer::DeltaResult GD::batchDelta(std::atomic<bool>& cond,
	const size_t start, const size_t cnt, const bool avg)
{
	Timer tmr;
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	double loss = 0.0;
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		auto p = pm->forward(pd->get(i));
		loss += pm->loss(p, pd->get(i).y);
		auto g = pm->backward(pd->get(i));
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
	}
	stat_t_grad_calc += tmr.elapseSd();
	tmr.restart();
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}
	stat_t_grad_post += tmr.elapseSd();
	return { i - start, i - start, move(grad) };
}

Trainer::DeltaResult GD::batchDelta(std::atomic<bool>& cond,
	const size_t start, const size_t cnt, const bool avg, const double adjust)
{
	Timer tmr;
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	double loss = 0.0;
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		Timer tt;
		auto p = pm->forward(pd->get(i));
		loss += pm->loss(p, pd->get(i).y);
		auto g = pm->backward(pd->get(i));
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
		double time = tt.elapseSd();
		this_thread::sleep_for(chrono::duration<double>(time*adjust));
	}
	stat_t_grad_calc += tmr.elapseSd();
	tmr.restart();
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}
	stat_t_grad_post += tmr.elapseSd();
	return { i - start, i - start, move(grad), loss };
}
