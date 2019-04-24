#include "PSGDwR.h"
#include "util/Timer.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include "util/Util.h"

using namespace std;

void PSGDwR::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		topRatio = stod(param[1]);
		global = param.size() > 2 ? param[2] == "global" : false; // true->global (gi*avg(g)), false->self (gi*gi)
		renewPortion = param.size() > 3 ? stod(param[3]) : 0.01;
		useRenewGrad = param.size() > 4 ? beTrueOption(param[4]) : false;
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for PSGDwR\n" + string(e.what()));
	}
	if(global)
		fp_cp = &PSGDwR::calcPriorityGlobal;
	else
		fp_cp = &PSGDwR::calcPrioritySelf;
}

std::string PSGDwR::name() const
{
	return "psgdr";
}

void PSGDwR::prepare()
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
	if(global){
		sumGrad.resize(paramWidth, 0.0);
	}
	renewSize = static_cast<size_t>(pd->size() * renewPortion);
	renewPointer = 0;
}

void PSGDwR::ready()
{
	// prepare gradient and priority
	if(global)
		gradient.reserve(pd->size());
	priority.resize(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		if(global){
			for(size_t j = 0; j < paramWidth; ++j)
				sumGrad[j] += g[j];
			gradient.push_back(move(g));
		} else{
			priority[i] = calcPriority(g);
		}
	}
	// prepare priority
	if(global){
		for(size_t i = 0; i < pd->size(); ++i){
			priority[i] = calcPriority(gradient[i]);
		}
	}
}

PSGDwR::~PSGDwR()
{
	LOG(INFO) << "[Stat-Trainer]: time-prio-pick: " << stat_t_prio_pick
		<< "\ttime-prio-update: " << stat_t_prio_update
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-renew: " << stat_t_grad_renew
		<< "\ttime-grad-post: " << stat_t_grad_post;
}

Trainer::DeltaResult PSGDwR::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	vector<double> grad(paramWidth, 0.0);
	// force renew the gradient of some data points
	size_t rcnt = renewSize + 1;
	while(--rcnt > 0){
		auto&& g = pm->gradient(pd->get(renewPointer));
		priority[renewPointer] = calcPriority(g);
		if(useRenewGrad){
			for(size_t j = 0; j < paramWidth; ++j)
				grad[j] += g[j];
		}
		if(global){
			updateSumGrad(renewPointer, g);
			gradient[renewPointer] = move(g);
		}
		renewPointer = (renewPointer + 1) % pd->size();
	}
	stat_t_grad_renew += tmr.elapseSd();
	// pick top-k
	tmr.restart();
	size_t k = static_cast<size_t>(round(topRatio*cnt));
	vector<int> topk = getTopK(start, end, k);
	stat_t_prio_pick += tmr.elapseSd();
	// update gradient and priority of data-points
	tmr.restart();
	for(auto i : topk){
		// calculate gradient
		auto&& g = pm->gradient(pd->get(i));
		stat_t_grad_calc += tmr.elapseSd();
		// calcualte priority
		tmr.restart();
		priority[i] = calcPriority(g);
		if(global){
			updateSumGrad(i, g);
		}
		stat_t_prio_update += tmr.elapseSd();
		// accumulate gradient result
		tmr.restart();
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		stat_t_grad_post += tmr.elapseSd();
		// final
		tmr.restart();
		if(global){
			gradient[i] = move(g);
		}
	}
	// calculate delta to report
	tmr.restart();
	double factor = -rate;
	if(avg)
		factor /= k;
	else
		factor *= static_cast<double>(cnt) / k;
	for(auto& v : grad)
		v *= factor;
	stat_t_grad_post += tmr.elapseSd();
	return { cnt, topk.size(), move(grad) };
}

Trainer::DeltaResult PSGDwR::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

float PSGDwR::calcPriority(const std::vector<double>& g)
{
	return (this->*fp_cp)(g);
}

float PSGDwR::calcPriorityGlobal(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), sumGrad.begin(), 0.0);
	return static_cast<float>(p);
}

float PSGDwR::calcPrioritySelf(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), g.begin(), 0.0);
	return static_cast<float>(p);
}

std::vector<int> PSGDwR::getTopK(const size_t first, const size_t last, const size_t k)
{
	std::vector<int> res;
	res.reserve(last - first);
	for(size_t i = first; i < last; ++i)
		res.push_back(static_cast<int>(i));
	if(res.size() <= k)
		return res;
	auto it = res.begin() + k;
	//partial_sort(res.begin(), it, res.end(),
	nth_element(res.begin(), it, res.end(),
		[&](const int l, const int r){
		return priority[l] > priority[r]; // pick the largest k
	});
	res.erase(it, res.end());
	return res;
}

void PSGDwR::updateSumGrad(const int i, const std::vector<double>& g){
	auto& go = gradient[i];
	for(size_t j = 0; j < paramWidth; ++j){
		sumGrad[j] += g[j] - go[j];
	}
}
