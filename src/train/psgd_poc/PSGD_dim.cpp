#include "PSGD_dim.h"
#include "util/Timer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

void PSGD_dim::init(const std::vector<std::string>& param)
{
	PSGD_poc::init(param);
}

std::string PSGD_dim::name() const
{
	return "psgd_poc_dim";
}

void PSGD_dim::ready()
{
	PSGD_poc::ready();
}

std::pair<size_t, std::vector<double>> PSGD_dim::batchDelta(
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
	vector<pair<int, int>> topk = getTopK(start, end, static_cast<size_t>(top_p*cnt*paramWidth));
	stat_t_priority += tmr.elapseSd();
	// calculate delta to report
	tmr.restart();
	vector<double> grad(paramWidth, 0.0);
	vector<int> dimCnt(paramWidth, 0);
	for(auto& p : topk){
		const int i = p.first;
		const int j = p.second;
		++dimCnt[j];
		grad[j] += gradient[i][j];
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

std::pair<size_t, std::vector<double>> PSGD_dim::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	return batchDelta(start, cnt, avg);
}

std::vector<std::pair<int, int>> PSGD_dim::getTopK(
	const size_t first, const size_t last, const size_t k)
{
	std::vector<std::pair<int, int>> res;
	res.reserve((last - first)*paramWidth);
	for(size_t i = first; i < last; ++i)
		for(size_t j = 0; j < paramWidth; ++j)
			res.emplace_back(static_cast<int>(i), static_cast<int>(j));
	if(res.size() <= k)
		return res;
	auto it = res.begin() + k;
	partial_sort(res.begin(), it, res.end(), 
		[&](const pair<int, int>& l, const pair<int, int>& r){
		return abs(gradient[l.first][l.second]) > abs(gradient[r.first][r.second]); // pick the largest k
	});
	res.erase(it, res.end());
	return res;
}
