#include "PSGD_point.h"
#include "util/Timer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

void PSGD_point::init(const std::vector<std::string>& param)
{
	PSGD_poc::init(param);
}

std::string PSGD_point::name() const
{
	return "psgd_poc_point";
}

void PSGD_point::ready()
{
	PSGD_poc::ready();
	priority.resize(pd->size());
}

std::pair<size_t, std::vector<double>> PSGD_point::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	size_t end = min(start + cnt, pd->size());
	Timer tmr;
	// calculate gradient of data-points
	updateGradient(start, end);
	stat_t_grad_calc += tmr.elapseSd();
	// dump gradient
	tmr.restart();
	dumpGradient();
	stat_t_grad_archive += tmr.elapseSd();
	// update priority and pick top-k
	tmr.restart();
	for(size_t i = 0; i < pd->size(); ++i){
		auto& g = gradient[i];
		auto p = inner_product(g.begin(), g.end(), g.begin(), 1.0);
		priority[i] = static_cast<float>(p);
	}
	vector<int> topk = getTopK(start, end, static_cast<size_t>(topRatio*cnt));
	stat_t_priority += tmr.elapseSd();
	// calculate delta to report
	tmr.restart();
	vector<double> grad(paramWidth, 0.0);
	for(int i : topk){
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += gradient[i][j];
	}
	double factor = -rate;
	if(avg)
		factor /= cnt;
	for(auto& v : grad)
		v *= factor;
	stat_t_grad_update += tmr.elapseSd();
	return make_pair(cnt, move(grad));
}

std::pair<size_t, std::vector<double>> PSGD_point::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

std::vector<int> PSGD_point::getTopK(const size_t first, const size_t last, const size_t k)
{
	std::vector<int> res;
	res.reserve(last - first);
	for(auto i = first; i < last; ++i)
		res.push_back(i);
	if(res.size() <= k)
		return res;
	auto it = res.begin() + k;
	partial_sort(res.begin(), it, res.end(), [&](const int l, const int r){
		return priority[l] > priority[r]; // pick the largest k
	});
	res.erase(it, res.end());
	return res;
}
