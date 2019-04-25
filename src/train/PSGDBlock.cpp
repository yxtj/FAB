#include "PSGDBlock.h"
#include "util/Util.h"
#include "util/Timer.h"
#include "logging/logging.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>

using namespace std;

void PSGDBlock::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		topRatio = stod(param[1]);
		dpblock = stoi(param[2]);
		renewRatio = param.size() > 3 ? stod(param[3]) : 0.01;
		useRenewGrad = param.size() > 4 ? beTrueOption(param[4]) : false;
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for BPSGD\n" + string(e.what()));
	}
}

std::string PSGDBlock::name() const
{
	return "bpsgd";
}

void PSGDBlock::prepare()
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
	// intialize block parameter
	if(dpblock == 0 || dpblock > pd->size())
		dpblock = pd->size();
	n_dpblock = (pd->size() + dpblock - 1) / dpblock;
	// intialize priority related parameter
	sumGrad.resize(paramWidth, 0.0);
	renewSize = static_cast<size_t>(pd->size() * renewRatio);
	renewPointer = 0;
}

void PSGDBlock::ready()
{
	// prepare gradient and priority
	// (Because we do not want to calculate gradient twice)
	gradient.reserve(n_dpblock);
	priority.resize(pd->size());
	vector<double> gblock(paramWidth, 0.0);
	for(size_t i = 0; i < pd->size(); ++i){
		if(i != 0 && i % dpblock == 0){
			for(auto& v : gblock)
				v /= static_cast<double>(dpblock);
			gradient.push_back(move(gblock));
			gblock.assign(paramWidth, 0.0);
		}
		auto g = pm->gradient(pd->get(i));
		for(size_t j = 0; j < paramWidth; ++j){
			sumGrad[j] += g[j]; // approximate
			gblock[j] += g[j];
		}
		float factor = static_cast<float>(n_dpblock) / i; // rectify the priority
		float p0 = calcPriority(g); // using current sumGrad
		priority[i] = p0 * factor * factor;
	}
	// rectify sumGrad
	const double factor = static_cast<double>(n_dpblock) / pd->size();
	for(auto& v : sumGrad)
		v *= factor;
	// process the last block
	size_t nlast = pd->size() - (n_dpblock - 1)*dpblock;
	for(auto& v : gblock)
		v /= static_cast<double>(nlast);
	gradient.push_back(move(gblock));
}

PSGDBlock::~PSGDBlock()
{
	LOG(INFO) << "[Stat-Trainer]: time-prio-pick: " << stat_t_prio_pick
		<< "\ttime-prio-update: " << stat_t_prio_update
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-renew: " << stat_t_grad_renew
		<< "\ttime-grad-post: " << stat_t_grad_post;
}

Trainer::DeltaResult PSGDBlock::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	vector<double> grad(paramWidth, 0.0);
	// force renew the gradient of some data points
	{
		vector<double> gbuf(paramWidth, 0.0);
		size_t bid_last = n_dpblock;
		int buf_cnt = 0;
		size_t rcnt = renewSize + 1;
		while(--rcnt > 0){
			auto&& g = pm->gradient(pd->get(renewPointer));
			priority[renewPointer] = calcPriority(g);
			if(useRenewGrad){
				for(size_t j = 0; j < paramWidth; ++j)
					grad[j] += g[j];
			}
			size_t bid = id2block(renewPointer);
			if(bid == bid_last){
				for(size_t j = 0; j < paramWidth; ++j)
					gbuf[j] += g[j];
				++buf_cnt;
			} else if(buf_cnt != 0){
				updateGrad(bid_last, gbuf, buf_cnt);
				bid_last = bid;
				gbuf = move(g);
				buf_cnt = 1;
			}
			renewPointer = (renewPointer + 1) % pd->size();
		}
		if(buf_cnt != 0){
			updateGrad(bid_last, gbuf, buf_cnt);
		}
	}
	stat_t_grad_renew += tmr.elapseSd();
	// pick top-k
	tmr.restart();
	size_t k = static_cast<size_t>(round(topRatio*cnt));
	vector<int> topk = getTopK(start, end, k);
	k = topk.size();
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
		updateGrad(id2block(i), g, 1);
		stat_t_prio_update += tmr.elapseSd();
		// accumulate gradient result
		tmr.restart();
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		stat_t_grad_post += tmr.elapseSd();
		// final
		tmr.restart();
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

Trainer::DeltaResult PSGDBlock::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

size_t PSGDBlock::id2block(size_t i) const
{
	return i / dpblock;
}

float PSGDBlock::calcPriority(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), sumGrad.begin(), 0.0);
	return static_cast<float>(p);
}

std::vector<int> PSGDBlock::getTopK(const size_t first, const size_t last, const size_t k)
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

void PSGDBlock::updateGrad(const size_t bid, const std::vector<double>& g, const int cnt){
	auto& go = gradient[bid];
	double factor = static_cast<double>(cnt) / dpblock;
	for(size_t j = 0; j < paramWidth; ++j){
		sumGrad[j] -= go[j];
		go[j] = go[j] * (1.0 - factor) + g[j] * factor;
		sumGrad[j] += go[j];
	}
}
