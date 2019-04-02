#include "GD.h"
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

void GD::ready()
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

std::pair<size_t, std::vector<double>> GD::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;
	for(i = start; i < end; ++i){
		auto g = pm->gradient(pd->get(i));
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
	return make_pair(i - start, move(grad));
}

std::pair<size_t, std::vector<double>> GD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;
	for(i = start; i < end && cond.load(); ++i){
		auto g = pm->gradient(pd->get(i));
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
	return make_pair(i - start, move(grad));
}
