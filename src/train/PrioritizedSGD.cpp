#include "PrioritizedSGD.h"
#include <algorithm>
#include <random>
#include <numeric>

using namespace std;

void PrioritizedSGD::init(const std::vector<std::string>& param)
{
	try{
		rate = stod(param[0]);
		dpblock = param.size() > 1 ? stoi(param[1]) : 1; // default: single data point
		dmblock = param.size() > 2 ? stoi(param[2]) : 0; // default: all dimension
	} catch(exception& e){
		throw invalid_argument("Cannot parse parameters for GD\n" + string(e.what()));
	}
}

std::string PrioritizedSGD::name() const
{
	return "psgd";
}

void PrioritizedSGD::ready()
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
	// intialize priority - batch parameter
	if(dpblock == 0 || dpblock > pd->size())
		dpblock = pd->size();
	n_dpblock = (pd->size() + dpblock - 1) / dpblock;
	if(dmblock == 0 || dmblock > paramWidth)
		dmblock = paramWidth;
	n_dmblock = (paramWidth + dmblock - 1) / dmblock;
	// intialize priority - priority queue
	uniform_real_distribution<float> dist(0.0, 1.0);
#if !defined(NDEBUG) || defined(_DEBUG) || defined(DEBUG)
	mt19937 gen(1);
#else
	random_device rdv;
	mt19937 gen(rdv);
#endif
	for(size_t i = 0; i < n_dpblock; ++i)
		for(size_t j = 0; j < n_dmblock; ++j)
			priQue.update(make_pair((int)i, (int)j), dist(gen));
}

std::pair<size_t, std::vector<double>> PrioritizedSGD::batchDelta(
	const size_t start, const size_t cnt, const bool avg)
{
	const size_t end = pd->size();
	vector<double> grad(paramWidth, 0.0);
	const size_t nblocks= cnt * n_dmblock / dpblock;
	// update
	size_t used_block = 0;
	size_t ndp = 0;
	unordered_set<int> used_dpblock;
	while(used_block++ < nblocks){
		pair<int, int> coord = priQue.top();
		//priQue.pop();
		if(used_dpblock.find(coord.first) != used_dpblock.end()){
			continue;
		}
		used_dpblock.insert(coord.first);
		size_t i_f = coord.first*dpblock, i_l = min(coord.first*dpblock + dpblock, end);
		//size_t j_f = coord.second*dbatch, j_l = min(coord.second*dbatch + dbatch, paramWidth);
		vector<double> tg(paramWidth, 0.0);
		size_t i;
		for(i = i_f; i < i_l; ++i){
			auto g = pm->gradient(pd->get(i));
			for(size_t j = 0; j < paramWidth; ++j)
				tg[j] += g[j];
			++ndp;
		}
		// update gradient
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += tg[j];
		// update priority
		vector<float> priority = calcPriority(tg);
		float factor = 1.0f / (i - i_f) / (i - i_f);
		for(int j = 0; j < n_dmblock; ++j)
			priQue.update(make_pair(coord.first, j), priority[j]);
	}
	if(ndp != 0){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= ndp;
		for(auto& v : grad)
			v *= factor;
	}
	return make_pair(cnt, move(grad));
}

std::pair<size_t, std::vector<double>> PrioritizedSGD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	const size_t end = pd->size();
	vector<double> grad(paramWidth, 0.0);
	const size_t nblocks = cnt * n_dmblock / dpblock;
	// update
	size_t used_block = 0;
	size_t ndp = 0;
	unordered_set<int> used_dpblock;
	while(cond.load() && used_block++ < nblocks){
		pair<int, int> coord = priQue.top();
		//priQue.pop();
		if(used_dpblock.find(coord.first) != used_dpblock.end()){
			continue;
		}
		used_dpblock.insert(coord.first);
		size_t i_f = coord.first*dpblock, i_l = min(coord.first*dpblock + dpblock, end);
		//size_t j_f = coord.second*dbatch, j_l = min(coord.second*dbatch + dbatch, paramWidth);
		vector<double> tg(paramWidth, 0.0);
		size_t i;
		for(i = i_f; cond.load() && i < i_l; ++i){
			auto g = pm->gradient(pd->get(i));
			for(size_t j = 0; j < paramWidth; ++j)
				tg[j] += g[j];
			++ndp;
		}
		// update gradient
		for(size_t j = 0; j < paramWidth; ++j)
			grad[j] += tg[j];
		// update priority
		vector<float> priority = calcPriority(tg);
		float factor = 1.0f / (i - i_f) / (i - i_f);
		for(int j = 0; j < n_dmblock; ++j)
			priQue.update(make_pair(coord.first, j), priority[j]*factor);
	}
	if(ndp != 0){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= ndp;
		for(auto& v : grad)
			v *= factor;
	}
	return make_pair(cnt, move(grad));
}

std::vector<float> PrioritizedSGD::calcPriority(const std::vector<double>& grad)
{
	vector<float> priority;
	size_t i = 0;
	while(i < paramWidth){
		double tp = 0.0;
		for(size_t j = 0; i<paramWidth && j < dmblock; ++j){
			tp += grad[i] * grad[i];
			++i;
		}
		priority.push_back(static_cast<float>(tp));
	}
	return priority;
}
