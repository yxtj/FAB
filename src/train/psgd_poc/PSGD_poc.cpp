#include "PSGD_poc.h"
#include "impl/TopKHolder.hpp"
#include "util/Timer.h"
#include "util/Util.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

void PSGD_poc::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		topRatio = stod(param[1]);
		global = param.size() > 2 ? param[2] == "global" : false; // true->global (gi*avg(g)), false->self (gi*gi)
		fname = param.size() > 3 ? param[3] : "";
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for PSGD\n" + string(e.what()));
	}
}

std::string PSGD_poc::name() const
{
	return "psgd_poc";
}

void PSGD_poc::ready()
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
	gradient.resize(pd->size(), vector<double>(paramWidth));
	if(global)
		sumGrad.resize(paramWidth, 0.0);
	if(!fname.empty())
		fout.open(fname + "-" + to_string(pd->partid()), ios::binary);
}

PSGD_poc::~PSGD_poc()
{
	LOG(INFO) << "[Stat]: time-prio-pick: " << stat_t_prio_pick
		<< "\ttime-prio-update: " << stat_t_prio_update
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-post: " << stat_t_grad_post
		<< "\ttime-grad-archive: " << stat_t_grad_archive;
}

void PSGD_poc::updateGradient(const size_t start, const size_t end)
{
	for(size_t i = start; i < end; ++i){
		auto g = pm->gradient(pd->get(i));
		if(global)
			updateSumGrad(i, g);
		gradient[i] = move(g);
	}
}

void PSGD_poc::updateSumGrad(const int i, const std::vector<double>& g){
	for(size_t j = 0; j < g.size(); ++j){
		sumGrad[j] += g[j] - gradient[i][j];
	}
}

void PSGD_poc::dumpGradient()
{
	if(fout.is_open()){
		for(auto& line : gradient){
			//fout.write((const char*)line.data(), line.size() * sizeof(double));
			for(auto& v : line){
				float f = static_cast<float>(v);
				fout.write((const char*)&f, sizeof(float));
			}
		}
	}
}
