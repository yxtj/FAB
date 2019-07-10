#include "PSGD_log.h"
#include "util/Timer.h"
#include "util/Util.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <limits>

using namespace std;

void PSGD_log::init(const std::vector<std::string>& param)
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

std::string PSGD_log::name() const
{
	return "psgd_poc_log";
}

void PSGD_log::prepare()
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
	prhd->init(pd->size());
	avgGrad.resize(paramWidth, 0.0);
	renewSize = static_cast<size_t>(pd->size() * renewRatio);
	topSize = static_cast<size_t>(pd->size() * topRatio);
	renewPointer = 0;
	pdump.init("priority-" + to_string(pd->partid()), true, true);
}

Trainer::DeltaResult PSGD_log::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	// dump priority
	pdump.dump(priority); // dump priority
	priority.clear();
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
	if(varAggReport){
		for(size_t j = 0; j < paramWidth; ++j)
			grad2[j] += grad1[j];
	}
	double factor = -rate;
	if(avg){
		if(varAggReport)
			factor = factor * 2 / (renewSize + topSize);
		else
			factor /= topSize;
	}
	for(auto& v : grad2)
		v *= factor;
	stat_t_post += tmr.elapseSd();
	return { cnt, topSize, move(grad2) };
}
