#include "GD.h"
using namespace std;

GD::GD()
	: rate(1.0)
{
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

std::pair<size_t, std::vector<double>> GD::batchDelta(const size_t start, const size_t cnt, const bool avg) const
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
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg) const
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
