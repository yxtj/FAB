#include "EM.h"
#include <exception>
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

std::pair<size_t, std::vector<double>> EM::batchDelta(
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
	return make_pair(i - start, move(grad));
}

std::pair<size_t, std::vector<double>> EM::batchDelta(
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
	return make_pair(i - start, move(grad));
}
