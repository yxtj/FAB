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
		//mergeDim = param.size() > 1 ? beTrueOption(param[1]) : true;
		//fullUpdate = param.size() > 2 ? beTrueOption(param[2]) : true;
		fname = param.size() > 2 ? param[2] : "";
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
		gradient[i] = move(g);
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
