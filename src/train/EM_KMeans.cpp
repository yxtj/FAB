#include "EM_KMeans.h"
#include <exception>
using namespace std;

void EM_KMeans::init(const std::vector<std::string>& param)
{
}

std::string EM_KMeans::name() const
{
	return "kmeans";
}

bool EM_KMeans::needAveragedDelta() const
{
	return false;
}

void EM_KMeans::prepare()
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

Trainer::DeltaResult EM_KMeans::batchDelta(
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
	return { i - start, i - start, move(grad) };
}

Trainer::DeltaResult EM_KMeans::batchDelta(
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
	return { i - start, i - start, move(grad) };
}

