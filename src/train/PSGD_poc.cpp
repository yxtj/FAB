#include "PSGD_poc.h"
#include "impl/TopKHolder.hpp"
#include "util/Timer.h"
#include "logging/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

void PSGD_poc::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		mergeDim = param.size() > 1 ? param[1] == "1" : true;
		fname = param.size() > 2 ? param[2] : "";
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for GD\n" + string(e.what()));
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
	if(!fname.empty())
		fout.open(fname + "-" + to_string(pd->partid()), ios::binary);
}

PSGD_poc::~PSGD_poc()
{
	LOG(INFO) << "[Stat]: time-priority: " << stat_t_priority
		<< "\ttime-archive: " << stat_t_archive
		<< "\ttime-grad: " << stat_t_grad;
}

std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	if(mergeDim){
		return batchDelta_point(start, cnt, avg);
	} else{
		return batchDelta_dim(start, cnt, avg);
	}
}

std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta_point(
	const size_t start, const size_t cnt, const bool avg)
{
	vector<double> grad(paramWidth, 0.0);
	// calculate all gradient and priority
	auto maintainTopK = [](vector<pair<float, int>>& priority_record, const size_t k){
		if(k >= priority_record.size())
			return;
		auto it_mid = priority_record.begin() + k;
		partial_sort(priority_record.begin(), it_mid, priority_record.end(),
			[](const pair<float, int>& l, const pair<float, int>& r){
			return l.first > r.first; // the larget the better
		});
		priority_record.erase(it_mid, priority_record.end());
	};
	Timer tmr;
	vector<vector<double>> gradient_buffer;
	gradient_buffer.reserve(pd->size());
	vector<pair<float, int>> priority_record;
	priority_record.reserve(pd->size());
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		auto p = inner_product(g.begin(), g.end(), g.begin(), 1.0);
		if(i%(4*cnt) == 0)
			maintainTopK(priority_record, cnt);
		priority_record.emplace_back(static_cast<float>(p), static_cast<int>(i));
		gradient_buffer.push_back(move(g));
	}
	maintainTopK(priority_record, cnt);
	stat_t_priority += tmr.elapseSd();
	// dump gradient
	tmr.restart();
	dumpGradient(gradient);
	stat_t_archive += tmr.elapseSd();
	// calculate gradient
	tmr.restart();
	for(auto& pc : priority_record){
		auto& g = gradient_buffer[pc.second];
		for(size_t i = 0; i < paramWidth; ++i)
			grad[i] += g[i];
	}
	gradient = move(gradient_buffer);
	double factor = -rate;
	if(avg)
		factor /= cnt;
	for(auto& v : grad)
		v *= factor;
	stat_t_grad += tmr.elapseSd();
	return make_pair(cnt, move(grad));
}

std::pair<size_t, std::vector<double>> PSGD_poc::batchDelta_dim(
	const size_t start, const size_t cnt, const bool avg)
{
	vector<double> grad(paramWidth, 0.0);
	// calculate all gradient and priority
	auto maintainTopK = [](vector<pair<float, pair<int, int>>>& priority_record, const size_t k){
		if(k >= priority_record.size())
			return;
		auto it_mid = priority_record.begin() + k;
		partial_sort(priority_record.begin(), it_mid, priority_record.end(),
			[](const pair<float, pair<int, int>>& l, const pair<float, pair<int, int>>& r){
			return l.first > r.first; // the larget the better
		});
		priority_record.erase(it_mid, priority_record.end());
	};
	Timer tmr;
	const size_t nblock = min(cnt, pd->size())*paramWidth;
	vector<vector<double>> gradient_buffer;
	gradient_buffer.reserve(pd->size());
	vector<pair<float, pair<int, int>>> priority_record;
	priority_record.reserve(4*nblock);
	for(size_t i = 0; i < pd->size(); ++i){
		auto g = pm->gradient(pd->get(i));
		if(i%(4*cnt) == 0)
			maintainTopK(priority_record, nblock);
		for(size_t j = 0; j < paramWidth; ++j){
			priority_record.emplace_back(static_cast<float>(abs(g[j])), make_pair((int)i, (int)j));
		}
		gradient_buffer.push_back(move(g));
	}
	/*vector<pair<double, pair<int, int>>> heap;
	heap.reserve(nblock+1);
	int end = static_cast<int>(min(cnt, pd->size()));
	for(int i = 0; i < end; ++i){
		for(int j = 0; j < static_cast<int>(paramWidth); ++j){
			heap.emplace_back(abs(gradient_buffer[i][j]), make_pair(i, j));
		}
	}
	make_heap(heap.begin(), heap.begin() + nblock,
		[](const pair<double, pair<int, int>>& l, const pair<double, pair<int, int>>& r){
		return l.first > r.first;
	}); // small top heap
	// heap[0] gives the smallest gradient
	for(int i = end; i < static_cast<int>(pd->size()); ++i){
		for(int j = 0; j < static_cast<int>(paramWidth); ++j){
			if(abs(gradient_buffer[i][j]) > heap[0].first){
				pop_heap(heap.begin(), heap.end());
				heap.pop_back();
				heap.emplace_back(abs(gradient_buffer[i][j]), make_pair(i, j));
				push_heap(heap.begin(), heap.end());
			}
		}
	}*/
	maintainTopK(priority_record, nblock);
	stat_t_priority += tmr.elapseSd();
	// dump gradient
	tmr.restart();
	dumpGradient(gradient);
	stat_t_archive += tmr.elapseSd();
	// calculate gradient
	tmr.restart();
	vector<int> dimCnt(paramWidth, 0);
	for(auto& pc : priority_record){
		auto c = pc.second;
		auto& g = gradient_buffer[c.first];
		int i = c.second;
		++dimCnt[i];
		grad[i] += g[i];
	}
	gradient = move(gradient_buffer);
	double factor = -rate;
	if(!avg)
		factor *= cnt;
	for(size_t i = 0; i < paramWidth; ++i){
		auto& v = grad[i];
		v *= factor / dimCnt[i];
	}
	stat_t_grad += tmr.elapseSd();
	return make_pair(cnt, move(grad));
}

void PSGD_poc::dumpGradient(const std::vector<std::vector<double>>& gradient)
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
