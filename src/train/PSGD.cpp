#include "PSGD.h"
#include "util/Timer.h"
#include "util/Util.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

using namespace std;

void PSGD::init(const std::vector<std::string>& param)
{
	try{
		do{
			rate = stod(param[0]);
			topRatio = stod(param[1]);
			renewRatio = stod(param[2]);
			if(!parsePriority(param[3], param[4]))
				throw invalid_argument("priority type is not recognized: " + param[3] + ", " + param[4]);
			//if(param.size() <= 6)
			//	break;
			//if(!parseGradient(param[5], param.size() > 6 ? param[6] : ""))
			//	throw invalid_argument("gradient type is not recognized: " + param[5]);
			if(param.size() <= 5)
				break; 
			if(!parseVariation(param[5]))
				throw invalid_argument("variation is not recognized: " + param[5]);
		} while(false);
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for PSGD\n" + string(e.what()));
	}
}

std::string PSGD::name() const
{
	return "psgd";
}

void PSGD::prepare()
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

void PSGD::ready()
{
	// prepare gradient (and priority)
	priority.resize(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		if(prioInitType == PriorityType::Projection){
			for(size_t j = 0; j < paramWidth; ++j)
				avgGrad[j] += g[j];
		} else{
			priority[i] = calcPriorityLength(g);
		}
		priorityIdx.push_back(static_cast<int>(i));
	}
	// prepare priority
	if(prioType == PriorityType::DecayExp){
		priorityWver.resize(pd->size(), 0);
		priorityDecayRate.resize(pd->size(), 1.0f);
	}
	if(prioInitType == PriorityType::Projection){
		for(size_t j = 0; j < paramWidth; ++j)
			avgGrad[j] /= pd->size();
		for(size_t i = 0; i < pd->size(); ++i){
			auto g = pm->gradient(pd->get(i));
			priority[i] = calcPriorityProjection(g);
		}
	}
	// prepare top priority list
	size_t k = static_cast<size_t>(pd->size() * topRatio);
	if(k == pd->size())
		--k;
	getTopK(k);
}

PSGD::~PSGD()
{
	LOG(INFO) << "[Stat-Trainer]: time-prio-pick: " << stat_t_prio_pick
		<< "\ttime-prio-update: " << stat_t_prio_update
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-renew: " << stat_t_grad_renew
		<< "\ttime-grad-post: " << stat_t_grad_post;
}

Trainer::DeltaResult PSGD::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	// phase 1: update the priority of some data points
	vector<double> grad1;
	size_t r;
	tie(r, grad1) = phaseUpdatePriority(renewSize);
	updateAvgGradDecay(grad1, static_cast<double>(r) / cnt);
	stat_t_grad_renew += tmr.elapseSd();
	// phase 2: calculate gradient for parameter
	tmr.restart();
	vector<double> grad2;
	size_t k = static_cast<size_t>(round(topRatio*cnt));
	tie(k, grad2) = phaseCalculateGradient(k);
	// variation
	if(varUpdateAvgGradTop){
		updateAvgGradDecay(grad2, static_cast<double>(k) / cnt);
	}
	// phase 3: post-process
	tmr.restart();
	wver += static_cast<unsigned>(r + k);
	if(varUpdateRptGradAll || varUpdateRptGradSel){
		for(size_t j = 0; j < paramWidth; ++j)
			grad2[j] += grad1[1];
	}
	double factor = -rate;
	if(avg)
		factor /= k;
	else
		factor *= static_cast<double>(cnt) / k;
	for(auto& v : grad2)
		v *= factor;
	stat_t_grad_post += tmr.elapseSd();
	return { cnt, k, move(grad2) };
}

Trainer::DeltaResult PSGD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

bool PSGD::parsePriority(const std::string & typeInit, const std::string & type)
{
	if(contains(typeInit, { "p","project","projection","g","global" })){
		prioInitType = PriorityType::Projection;
	} else if(contains(typeInit, { "l","length","s","square","self" })){
		prioInitType = PriorityType::Length;
	} else
		return false;
	if(contains(type, { "p","project","projection","g","global" })){
		prioType = PriorityType::Projection;
	} else if(contains(type, { "l","length","s","square","self" })){
		prioType = PriorityType::Length;
	} else if(contains(type, { "d","e","decayexp" })){
		prioType = PriorityType::DecayExp;
	} else
		return false;
	if(prioType == PriorityType::Length)
		fp_cp = &PSGD::calcPriorityLength;
	else
		fp_cp = &PSGD::calcPriorityProjection;
	//if(!factor.empty())
	//	prioDecayFactor = stod(factor);
	return true;
}

bool PSGD::parseGradient(const std::string & type, const std::string & factor)
{
	//if(contains(type, { "a","accurate","incr","increament","increamental" }))
	//	gradType = GradientType::Increment;
	//else if(contains(type, { "d","decay" }))
	//	gradType = GradientType::Decay;
	//else
	//	return false;
	//if(!factor.empty())
	//	gradDecayFactor = stod(factor);
	return true;
}

bool PSGD::parseVariation(const std::string & str)
{
	if(str.find('j') != string::npos || str.find("rj") != string::npos)
		varUpdateRptGradAll = true;
	if(str.find('s') != string::npos || str.find("rs") != string::npos)
		varUpdateRptGradSel = true;
	if(str.find('t') != string::npos || str.find("at") != string::npos)
		varUpdateAvgGradTop= true;
	return true;
}

float PSGD::calcPriority(const std::vector<double>& g)
{
	return (this->*fp_cp)(g);
}

float PSGD::calcPriorityProjection(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), avgGrad.begin(), 0.0);
	return static_cast<float>(p);
}

float PSGD::calcPriorityLength(const std::vector<double>& g)
{
	auto p = inner_product(g.begin(), g.end(), g.begin(), 0.0);
	return static_cast<float>(p);
}

void PSGD::updateAvgGradDecay(const std::vector<double>& g, const double f)
{
	for(size_t j = 0; j < paramWidth; ++j){
		avgGrad[j] = (1 - f)*avgGrad[j] * f + f * g[j];
	}
}

std::pair<size_t, std::vector<double>> PSGD::phaseUpdatePriority(const size_t r)
{
	vector<double> grad(paramWidth, 0.0);
	size_t n = 0;
	// force renew the gradient of some data points
	size_t rcnt = r + 1;
	while(--rcnt > 0){
		auto&& g = pm->gradient(pd->get(renewPointer));
		float p = calcPriority(g);
		if(prioType == PriorityType::DecayExp){
			updatePriorityDecay(p, renewPointer);
		}
		priority[renewPointer] = p;
		if(varUpdateRptGradAll || (varUpdateRptGradSel && p >= prioThreshold)){
			for(size_t j = 0; j < paramWidth; ++j)
				grad[j] += g[j];
		}
		renewPointer = (renewPointer + 1) % pd->size();
	}
	return make_pair(move(n), move(grad));
}

std::pair<size_t, std::vector<double>> PSGD::phaseCalculateGradient(const size_t k)
{
	Timer tmr;
	vector<double> grad(paramWidth, 0.0);
	vector<int> topk;
	if(prioType == PriorityType::DecayExp)
		getTopKDecay(k);
	else
		getTopK(k);
	stat_t_prio_pick += tmr.elapseSd();
	// update gradient and priority of data-points
	for(size_t i = 0; i < k; ++i){
		size_t id = priorityIdx[i];
		// calculate gradient
		tmr.restart();
		auto&& g = pm->gradient(pd->get(id));
		stat_t_grad_calc += tmr.elapseSd();
		// calcualte priority
		tmr.restart();
		float p = calcPriority(g);
		if(prioType == PriorityType::DecayExp){
			updatePriorityDecay(p, id);
		}
		priority[id] = p;
		stat_t_prio_update += tmr.elapseSd();
		// accumulate gradient result
		tmr.restart();
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		stat_t_grad_post += tmr.elapseSd();
	}
	return make_pair(k, move(grad));
}

void PSGD::getTopK(const size_t k)
{
	if(priorityIdx.size() <= k){
		prioThreshold = numeric_limits<float>::lowest();
		return;
	}
	auto it = priorityIdx.begin() + k;
	//partial_sort(res.begin(), it, res.end(),
	nth_element(priorityIdx.begin(), it, priorityIdx.end(),
		[&](const int l, const int r){
		return priority[l] > priority[r]; // pick the largest k
	});
	prioThreshold = priority[*it];
}

void PSGD::getTopKDecay(const size_t k)
{
	if(priorityIdx.size() <= k){
		prioThreshold = numeric_limits<float>::lowest();
		return;
	}
	auto it = priorityIdx.begin() + k;
	//partial_sort(res.begin(), it, res.end(),
	nth_element(priorityIdx.begin(), it, priorityIdx.end(),
		[&](const int l, const int r){
		return priority[l] * pow(priorityDecayRate[l], priorityWver[l] - wver)
			> priority[r] * pow(priorityDecayRate[r], priorityWver[r] - wver); // pick the largest k
	});
	prioThreshold = priority[*it];
}

void PSGD::updatePriorityDecay(float p, size_t id)
{
	const float pold = priority[id];
	if(pold == 0.0f){
		priorityDecayRate[id] = 1.0f;
	} else if(wver == priorityWver[id]){
		priorityDecayRate[id] = priorityDecayRate[id];
	} else{
		float dp = pold - p;
		size_t dn = wver - priorityWver[id];
		// dp/p = 1 - exp(lambda * dn)
		priorityDecayRate[id] = logf(1 - dp / p) / dn;
	}
	priorityWver[id] = wver;
}
