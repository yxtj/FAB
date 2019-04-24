#include "PSGDDecay.h"
#include "util/Timer.h"
#include "util/Util.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>

using namespace std;

void PSGDDecay::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		topRatio = stod(param[1]);
		decay = stod(param[2]);
		renewRatio= param.size() > 3 ? stod(param[3]) : 0.01;
		useRenewGrad = param.size() > 4 ? beTrueOption(param[4]) : false;
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for PSGDDecay\n" + string(e.what()));
	}
}

std::string PSGDDecay::name() const
{
	return "psgdd";
}

void PSGDDecay::prepare()
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
	avgGrad.resize(paramWidth, 0.0);
	renewSize = static_cast<size_t>(pd->size() * renewRatio);
	renewPointer = 0;
}

void PSGDDecay::ready()
{
	// prepare gradient and priority
	// (Because we do not want to calculate gradient twice)
	priority.resize(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		for(size_t j = 0; j < paramWidth; ++j){
			avgGrad[j] += g[j];
		}
		float p0 = calcPriority(g); // using current avgGrad
		// rectify the priority
		priority[i] = p0 / (i * i);
	}
}

PSGDDecay::~PSGDDecay()
{
	LOG(INFO) << "[Stat-Trainer]: time-prio-pick: " << stat_t_prio_pick
		<< "\ttime-prio-update: " << stat_t_prio_update
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-renew: " << stat_t_grad_renew
		<< "\ttime-grad-post: " << stat_t_grad_post;
}

Trainer::DeltaResult PSGDDecay::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	vector<double> grad;
	// force renew the gradient of some data points
	vector<double> gbuf(paramWidth, 0.0);
	size_t rcnt = renewSize + 1;
	while(--rcnt > 0){
		auto&& g = pm->gradient(pd->get(renewPointer));
		priority[renewPointer] = calcPriority(g);
		for(size_t j = 0; j < paramWidth; ++j)
			gbuf[j] += g[j];
		renewPointer = (renewPointer + 1) % pd->size();
	}
	updateAvgGrad(gbuf);
	if(useRenewGrad){
		grad = move(gbuf);
		gbuf.assign(paramWidth, 0.0);
	} else{
		grad.assign(paramWidth, 0.0);
	}
	// at this time: gbuf is empty
	stat_t_grad_renew += tmr.elapseSd();
	// pick top-k
	tmr.restart();
	size_t k = static_cast<size_t>(round(topRatio*cnt));
	vector<int> topk = getTopK(start, end, k);
	k = topk.size();
	stat_t_prio_pick += tmr.elapseSd();
	// update gradient and priority of data-points
	tmr.restart();
	const double dfactor = decay / k;
	for(auto i : topk){
		// calculate gradient
		auto&& g = pm->gradient(pd->get(i));
		stat_t_grad_calc += tmr.elapseSd();
		// calcualte priority
		tmr.restart();
		priority[i] = calcPriority(g);
		//updateAvgGrad(g);
		updateAvgGrad(g, dfactor);
		stat_t_prio_update += tmr.elapseSd();
		// accumulate gradient result
		tmr.restart();
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		stat_t_grad_post += tmr.elapseSd();
		// final
		tmr.restart();
	}
	//updateAvgGrad(grad);
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

Trainer::DeltaResult PSGDDecay::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

float PSGDDecay::calcPriority(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), avgGrad.begin(), 0.0);
	return static_cast<float>(p);
}

std::vector<int> PSGDDecay::getTopK(const size_t first, const size_t last, const size_t k)
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

void PSGDDecay::updateAvgGrad(const std::vector<double>& g){
	updateAvgGrad(g, decay);
}

void PSGDDecay::updateAvgGrad(const std::vector<double>& g, const double f)
{
	for(size_t j = 0; j < paramWidth; ++j){
		avgGrad[j] = avgGrad[j] * f + g[j] * (1.0 - f);
	}
}
