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
	topSize = static_cast<size_t>(pd->size() * topRatio);
	renewPointer = 0;
}

void PSGD::ready()
{
	// prepare initialize priority
	priority.resize(pd->size());
	priorityIdx.resize(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		if(prioInitType == PriorityType::Length){
			priority[i] = calcPriorityLength(g);
		}
		if(prioInitType != PriorityType::Length && prioType != PriorityType::Length){
			for(size_t j = 0; j < paramWidth; ++j)
				avgGrad[j] += g[j];
		}
		priorityIdx[i] = static_cast<int>(i);
	}
	if(prioInitType != PriorityType::Length && prioType != PriorityType::Length){
		for(size_t j = 0; j < paramWidth; ++j)
			avgGrad[j] /= pd->size();
	}
	if(prioInitType == PriorityType::Projection){
		for(size_t i = 0; i < pd->size(); ++i){
			auto g = pm->gradient(pd->get(i));
			priority[i] = calcPriorityProjection(g);
		}
	}
	// prepare for runtime priority
	if(prioType == PriorityType::DecayExp){
		priorityWver.resize(pd->size(), 0);
		priorityDecayRate.resize(pd->size(), 1.0f);
	}
	// prepare top priority list
	getTopK(topSize);
}

PSGD::~PSGD()
{
	LOG(INFO) << "[Stat-Trainer]: "
		<< "renew-phase: " << stat_t_renew << "\t"
		<< "update-phase: " << stat_t_update << "\t"
		<< "post-phase: " << stat_t_post << "\t"
		<< "pick-topk: " << stat_t_topk<< "\t"
		<< "calc-gradient: " << stat_t_grad_calc << "\t"
		<< "calc-priority: " << stat_t_prio_calc;
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
	updateAvgGrad(grad1, static_cast<double>(r) / cnt);
	stat_t_renew += tmr.elapseSd();
	// phase 2: calculate gradient for parameter
	tmr.restart();
	vector<double> grad2;
	size_t k;
	tie(k, grad2) = phaseCalculateGradient(topSize);
	// variation
	if(varUpdateAvgGradTop){
		updateAvgGrad(grad2, static_cast<double>(k) / cnt);
	}
	stat_t_update += tmr.elapseSd();
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
	for(auto& v : grad2)
		v *= factor;
	stat_t_post += tmr.elapseSd();
	return { cnt, k, move(grad2) };
}

Trainer::DeltaResult PSGD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
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
		updatePriority(p, renewPointer);
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
	getTopK(k);
	stat_t_topk += tmr.elapseSd();
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
		updatePriority(p, renewPointer);
		stat_t_prio_calc += tmr.elapseSd();
		// accumulate gradient result
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
	}
	return make_pair(k, move(grad));
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
	if(prioType == PriorityType::Length){
		fp_cp = &PSGD::calcPriorityLength;
		fp_up = &PSGD::updatePriorityKeep;
		fp_pkp = &PSGD::topKPredKeep;
	} else if(prioType == PriorityType::Projection){
		fp_cp = &PSGD::calcPriorityProjection;
		fp_up = &PSGD::updatePriorityKeep;
		fp_pkp = &PSGD::topKPredKeep;
	} else{
		fp_cp = &PSGD::calcPriorityProjection;
		fp_up = &PSGD::updatePriorityDecay;
		fp_pkp = &PSGD::topKPredDecay;
	}
	//if(!factor.empty())
	//	prioDecayFactor = stod(factor);
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

void PSGD::updatePriority(float p, size_t id)
{
	return (this->*fp_up)(p, id);
}

void PSGD::updatePriorityKeep(float p, size_t id)
{
	priority[id] = p;
}

void PSGD::updatePriorityDecay(float p, size_t id)
{
	const float pold = priority[id];
	if(wver == priorityWver[id] || pold == 0.0f || p == 0.0f){
		priorityDecayRate[id] = priorityDecayRate[id];
	} else{
		float dp = p / pold;
		if(dp <= 0.0f){
			priorityDecayRate[id] = 1.0f;
		} else{
			size_t dn = wver - priorityWver[id];
			// dp/p = 1 - exp(lambda * dn)
			// lambda = ln(1-dp/p)/dn
			priorityDecayRate[id] = -logf(dp) / dn;
		}
	}
	priorityWver[id] = wver;
	priority[id] = p;
}

void PSGD::updateAvgGrad(const std::vector<double>& g, const double f)
{
	for(size_t j = 0; j < paramWidth; ++j){
		avgGrad[j] = (1 - f)*avgGrad[j] * f + f * g[j];
	}
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
		bind(fp_pkp, this, placeholders::_1, placeholders::_2));
	prioThreshold = priority[*it];
}

bool PSGD::topKPredKeep(const int l, const int r)
{
	return priority[l] > priority[r]; // pick the largest k
}

bool PSGD::topKPredDecay(const int l, const int r)
{
	return priority[l] * exp(priorityDecayRate[l] * (wver - priorityWver[l]))
		> priority[r] * exp(priorityDecayRate[r] * (wver - priorityWver[r]));
}
