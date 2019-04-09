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
		top_p = stod(param[1]);
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
	LOG(INFO) << "[Stat]: time-priority: " << stat_t_priority
		<< "\ttime-grad-calc: " << stat_t_grad_calc
		<< "\ttime-grad-archive: " << stat_t_grad_archive
		<< "\ttime-grad-update: " << stat_t_grad_update;
}


std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta_point(
	const size_t start, const size_t cnt, const bool avg)
{
	static auto maintainTopK = [](vector<pair<float, int>>& priority_record, const size_t k){
		if(k >= priority_record.size())
			return;
		auto it_mid = priority_record.begin() + k;
		partial_sort(priority_record.begin(), it_mid, priority_record.end(),
			[](const pair<float, int>& l, const pair<float, int>& r){
			return l.first > r.first; // the larget the better
		});
		priority_record.erase(it_mid, priority_record.end());
	};
	// update gradient of data points
	size_t end = min(start + cnt, pd->size());
	updateGradient(start, end);
	// update priority and pick top-k
	Timer tmr;
	vector<pair<float, int>> priority_record;
	priority_record.reserve(pd->size());
	for(size_t i = start; i < end; ++i){
		if(i%(4*cnt) == 0)
			maintainTopK(priority_record, cnt);
		auto& g = gradient[i];
		auto p = inner_product(g.begin(), g.end(), g.begin(), 1.0);
		priority_record.emplace_back(static_cast<float>(p), static_cast<int>(i));
	}
	maintainTopK(priority_record, cnt);
	stat_t_priority += tmr.elapseSd();
	// calculate delta to report
	tmr.restart();
	vector<double> grad(paramWidth, 0.0);
	for(auto& pc : priority_record){
		auto& g = gradient[pc.second];
		for(size_t i = 0; i < paramWidth; ++i)
			grad[i] += g[i];
	}
	double factor = -rate;
	if(avg)
		factor /= cnt;
	for(auto& v : grad)
		v *= factor;
	stat_t_grad_update += tmr.elapseSd();
	return make_pair(cnt, move(grad));
}

std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta_dim(
	const size_t start, const size_t cnt, const bool avg)
{
	static auto maintainTopK = [](vector<pair<float, pair<int, int>>>& priority_record, const size_t k){
		if(k >= priority_record.size())
			return;
		auto it_mid = priority_record.begin() + k;
		partial_sort(priority_record.begin(), it_mid, priority_record.end(),
			[](const pair<float, pair<int, int>>& l, const pair<float, pair<int, int>>& r){
			return l.first > r.first; // the larget the better
		});
		priority_record.erase(it_mid, priority_record.end());
	};
	// update gradient of data points
	size_t end = min(start + cnt, pd->size());
	updateGradient(start, end);
	// update priority and pick top-k
	Timer tmr;
	const size_t nblock = min(cnt, pd->size())*paramWidth;
	vector<pair<float, pair<int, int>>> priority_record;
	priority_record.reserve(4*nblock);
	for(size_t i = start; i < end; ++i){
		if(i%(4*cnt) == 0)
			maintainTopK(priority_record, nblock);
		auto& g = gradient[i];
		for(size_t j = 0; j < paramWidth; ++j){
			priority_record.emplace_back(static_cast<float>(abs(g[j])), make_pair((int)i, (int)j));
		}
	}
	maintainTopK(priority_record, nblock);
	stat_t_priority += tmr.elapseSd();
	tmr.restart();
	// calculate gradient
	tmr.restart();
	vector<double> grad(paramWidth, 0.0);
	vector<int> dimCnt(paramWidth, 0);
	for(auto& pc : priority_record){
		auto c = pc.second;
		auto& g = gradient[c.first];
		int i = c.second;
		++dimCnt[i];
		grad[i] += g[i];
	}
	double factor = -rate;
	if(!avg)
		factor *= cnt;
	for(size_t i = 0; i < paramWidth; ++i){
		auto& v = grad[i];
		v *= factor / dimCnt[i];
	}
	stat_t_grad_update += tmr.elapseSd();
	return make_pair(cnt, move(grad));
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
