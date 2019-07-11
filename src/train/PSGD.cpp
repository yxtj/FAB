#include "PSGD.h"
#include "util/Timer.h"
#include "util/Util.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <limits>

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
			if(!parseDecay(param[5]))
				throw invalid_argument("decay type is not recognized: " + param[5]);
			if(param.size() <= 6)
				break; 
			if(!parseVariation(param[6]))
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
	paramWidth = pm->paramWidth();
	if(pm->getKernel()->needInitParameterByData()){
		// initialize parameter by data
		Parameter p;
		p.init(paramWidth, 0.0);
		pm->setParameter(p);
		size_t s = pd->size();
		for(size_t i = 0; i < s; ++i){
			pm->getKernel()->initVariables(
				pd->get(i).x, pm->getParameter().weights, pd->get(i).y, nullptr);
		}
	}
	// local paraemters
	priority.resize(pd->size());
	prhd->init(pd->size());
	avgGrad.resize(paramWidth, 0.0);
	renewSize = static_cast<size_t>(pd->size() * renewRatio);
	topSize = static_cast<size_t>(pd->size() * topRatio);
	renewPointer = 0;
}

void PSGD::ready()
{
	// prepare initialize priority
	prhd->init(pd->size());
	priorityIdx.resize(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		if(prioInitType == PriorityType::Length){
			float p = calcPriorityLength(g);
			prhd->set(i, 0, p);
		}
		if(prioInitType != PriorityType::Length || prioType != PriorityType::Length){
			for(size_t j = 0; j < paramWidth; ++j)
				avgGrad[j] += g[j];
		}
		priorityIdx[i] = static_cast<int>(i);
	}
	if(prioInitType != PriorityType::Length || prioType != PriorityType::Length){
		for(size_t j = 0; j < paramWidth; ++j)
			avgGrad[j] /= pd->size();
	}
	if(prioInitType == PriorityType::Projection){
		for(size_t i = 0; i < pd->size(); ++i){
			auto g = pm->gradient(pd->get(i));
			float p = calcPriorityProjection(g);
			prhd->set(i, 0, p);
		}
	}
	// prepare top priority list
	getTopK(topSize);
	moveWver();
}

PSGD::~PSGD()
{
	LOG(INFO) << "[Stat-Trainer]: "
		<< "phase-renew: " << stat_t_renew << "\t"
		<< "phase-update: " << stat_t_update << "\t"
		<< "phase-post: " << stat_t_post << "\t"
		<< "u-topk: " << stat_t_u_topk << "\t"
		<< "u-gradient: " << stat_t_u_grad << "\t"
		<< "u-priority: " << stat_t_u_prio << "\t"
		<< "u-merge: " << stat_t_u_merge;
	delete prhd;
	prhd = nullptr;
}

Trainer::DeltaResult PSGD::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	// phase 1: update the priority of some data points
	vector<double> grad1 = phaseUpdatePriority(renewSize);
	updateAvgGrad(grad1, static_cast<double>(renewSize) / cnt);
	stat_t_renew += tmr.elapseSd();
	// phase 2: calculate gradient for parameter
	tmr.restart();
	vector<double> grad2 = phaseCalculateGradient(topSize);
	// variation
	if(varAggAverage){
		updateAvgGrad(grad2, static_cast<double>(topSize) / cnt);
	}
	stat_t_update += tmr.elapseSd();
	// phase 3: post-process
	tmr.restart();
	moveWver();
	const double factor = -rate;
	if(avg){
		if(varAggReport){
			for(size_t j = 0; j < paramWidth; ++j)
				grad2[j] = factor * (grad1[j] / renewSize + grad2[j] / topSize);
		} else{
			for(size_t j = 0; j < paramWidth; ++j)
				grad2[j] *= factor / topSize;
		}
	} else{
		if(varAggReport){
			for(size_t j = 0; j < paramWidth; ++j)
				grad2[j] += grad1[j];
		}
	}
	stat_t_post += tmr.elapseSd();
	return { cnt, topSize, move(grad2) };
}

Trainer::DeltaResult PSGD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

std::vector<double> PSGD::phaseUpdatePriority(const size_t r)
{
	vector<double> grad(paramWidth, 0.0);
	// force renew the gradient of some data points
	size_t rcnt = r + 1;
	while(--rcnt > 0){
		auto&& g = pm->gradient(pd->get(renewPointer));
		float p = calcPriority(g);
		prhd->update(renewPointer, wver, p);
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		renewPointer = (renewPointer + 1) % pd->size();
	}
	return grad;
}

std::vector<double> PSGD::phaseCalculateGradient(const size_t k)
{
	Timer tmr;
	vector<double> grad(paramWidth, 0.0);
	getTopK(k);
	stat_t_u_topk += tmr.elapseSd();
	// update gradient and priority of data-points
	for(size_t i = 0; i < k; ++i){
		size_t id = priorityIdx[i];
		// calculate gradient
		tmr.restart();
		auto&& g = pm->gradient(pd->get(id));
		stat_t_u_grad += tmr.elapseSd();
		// calcualte priority
		if(varAggLearn){
			tmr.restart();
			float p = calcPriority(g);
			prhd->update(i, wver, p);
			stat_t_u_prio += tmr.elapseSd();
		}
		// accumulate gradient result
		tmr.restart();
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += g[j];
		stat_t_u_merge += tmr.elapseSd();
	}
	return grad;
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
	} else
		return false;
	if(prioType == PriorityType::Length){
		fp_cp = &PSGD::calcPriorityLength;
	} else if(prioType == PriorityType::Projection){
		fp_cp = &PSGD::calcPriorityProjection;
	}
	//if(!factor.empty())
	//	prioDecayFactor = stod(factor);
	return true;
}

bool PSGD::parseDecay(const std::string & type)
{
	if(prhd != nullptr){
		delete prhd;
		prhd = nullptr;
	}
	if(type.empty() || contains(type, { "k","keep" })){
		prhd = new PriorityHolderKeep();
	} else if(contains(type, { "e","d","l","linear","1" })){
		prhd = new PriorityHolderExpLinear();
	} else if(contains(type, { "q","quadratic","2" })){
		prhd = new PriorityHolderExpQuadratic();
	}
	return prhd != nullptr;
}

bool PSGD::parseVariation(const std::string & str)
{
	if(str.find('r') != string::npos)
		varAggReport = true;
	if(str.find('l') != string::npos)
		varAggLearn = true;
	if(str.find('a') != string::npos)
		varAggAverage = true;
	if(str.find('p') != string::npos || str.find('d') != string::npos)
		varVerDP = true;
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

void PSGD::getTopK(const size_t k)
{
	if(priorityIdx.size() <= k){
		return;
	}
	for(size_t i = 0; i < pd->size(); ++i)
		priority[i] = prhd->get(i, wver);
	auto it = priorityIdx.begin() + k;
	//partial_sort(res.begin(), it, res.end(),
	nth_element(priorityIdx.begin(), it, priorityIdx.end(), 
		[&](const int l, const int r){
		return priority[l] > priority[r];
	});
}

void PSGD::moveWver()
{
	if(varVerDP){
		wver += static_cast<unsigned>((renewSize + topSize) / 32);
	} else{
		wver += 1;
	}
}

void PSGD::updateAvgGrad(const std::vector<double>& g, const double f)
{
	for(size_t j = 0; j < paramWidth; ++j){
		avgGrad[j] = (1 - f)*avgGrad[j] * f + f * g[j];
	}
}
