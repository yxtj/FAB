#include "PSGD.h"
#include "util/Timer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

void PSGD::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		topRatio = stod(param[1]);
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for PSGD\n" + string(e.what()));
	}
}

std::string PSGD::name() const
{
	return "psgd";
}

void PSGD::ready()
{
	// initialize parameter by data
	paramWidth = pm->paramWidth();
	if(pm->getKernel()->needInitParameterByData()){
		Parameter p;
		p.init(paramWidth, 0.0);
		pm->setParameter(p);
		size_t s = pd->size();
		for(size_t i = 0; i < s; ++i){
			pm->getKernel()->initVariables(
				pd->get(i).x, pm->getParameter().weights, pd->get(i).y, nullptr);
		}
	}
	// prepare gradient
	gradient.resize(pd->size(), vector<double>(paramWidth));
	// prepare priority
	priority.reserve(pd->size());
}

PSGD::~PSGD()
{
	LOG(INFO) << "[Stat]: time-priority: " << stat_t_priority
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-aggr: " << stat_t_grad_aggr;
}

std::pair<size_t, std::vector<double>> PSGD::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	// update priority and pick top-k
	vector<int> topk = getTopK(start, end, static_cast<size_t>(top_p*cnt));
	stat_t_priority += tmr.elapseSd();
	// calculate gradient of data-points
	tmr.restart();
	updateGnP(topk);
	stat_t_grad_calc += tmr.elapseSd();
	// calculate delta to report
	tmr.restart();
	vector<double> grad(paramWidth, 0.0);
	for(int i : topk){
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += gradient[i][j];
	}
	double factor = -rate;
	if(avg)
		factor /= cnt;
	for(auto& v : grad)
		v *= factor;
	stat_t_grad_aggr += tmr.elapseSd();
	return make_pair(cnt, move(grad));
}

std::pair<size_t, std::vector<double>> PSGD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

std::vector<int> PSGD::getTopK(const size_t first, const size_t last, const size_t k)
{
	std::vector<int> res;
	res.reserve(last - first);
	for(auto i = first; i < last; ++i)
		res.push_back(i);
	if(res.size() <= k)
		return res;
	auto it = res.begin() + k;
	partial_sort(res.begin(), it, res.end(), [&](const int l, const int r){
		return priority[l] > priority[r]; // pick the largest k
	});
	res.erase(it, res.end());
	return res;
}

void PSGD::updateGnP(const std::vector<int>& topk)
{
	for(auto i : topk){
		auto&& g = pm->gradient(pd->get(i));
		auto p = inner_product(g.begin(), g.end(), g.end(), 1.0);
		priority[i] = static_cast<float>(p);
		gradient[i] = move(g);
	}
}
