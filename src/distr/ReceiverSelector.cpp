#include "ReceiverSelector.h"
#include <random>
#include <algorithm>

using namespace std;

void ReceiverSelector::init(const std::vector<std::string>& param, const size_t nWorker)
{
	nw = nWorker;
}

// ---- broadcast ----

struct ReceiverSelectorBroadcast : public ReceiverSelector{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker){
		ReceiverSelector::init(param, nWorker);
		all.reserve(nw);
		for(int i = 0; i < nw; ++i)
			all.push_back(i);
	}
	virtual std::vector<int> getTargets(const int sourceId) {
		return all;
	}
private:
	std::vector<int> all;
};

// ---- ring ----

struct ReceiverSelectorRing: public ReceiverSelector{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker){
		ReceiverSelector::init(param, nWorker);
		k = stoi(param[1]);
		mlist.resize(nw);
		for(int i = 0; i < nw; ++i){
			for(int j = 0; j < k; ++j)
				mlist[i].push_back((i + j) % nw);
		}
	}
	virtual std::vector<int> getTargets(const int sourceId) {
		return mlist[sourceId];
	}
private:
	int k;
	std::vector<std::vector<int>> mlist;
};

// ---- random ----

struct ReceiverSelectorRandom: public ReceiverSelector{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker){
		ReceiverSelector::init(param, nWorker);
		k = stoi(param[1]);
		auto seed = param.size() > 2 ? stoul(param[2]) : (unsigned long)1;
		gen.seed(seed);
		dist = uniform_int_distribution<int>(0, nWorker);
	}
	virtual std::vector<int> getTargets(const int sourceId) {
		vector<int> res(k);
		vector<bool> used(k, false);
		for(int i = 0; i < k; ++i){
			int v = dist(gen);
			while(used[v])
				v = (v + 1) % nw;
			res[i] = v;
			used[v] = true;
		}
		return res;
	}
private:
	int k;
	uniform_int_distribution<int> dist;
	mt19937 gen;
};

// ---- hash ----

struct ReceiverSelectorHash: public ReceiverSelector{
	virtual void init(const std::vector<std::string>& param, const size_t nWorker){
		ReceiverSelector::init(param, nWorker);
		int k = stoi(param[1]);
		mlist.resize(nw);
		for(int i = 0; i < nw; ++i){
			int fac = FACTOR[i%FACTOR.size()];
			int off = i / FACTOR.size();
			for(int j = 0; j < k; ++j){
				mlist[i].push_back(((j + 1)*fac + off) % nw);
			}
			sort(mlist[i].begin(), mlist[i].end());
			auto it = unique(mlist[i].begin(), mlist[i].end());
			mlist[i].erase(it, mlist[i].end());
		}
	}
	virtual std::vector<int> getTargets(const int sourceId) {
		return mlist[sourceId];
	}
private:
	static std::vector<int> FACTOR; // primes without 2 and 5
	std::vector<std::vector<int>> mlist;
};
std::vector<int> ReceiverSelectorHash::FACTOR =
	{ 3,7,11,13,17,19,23,29,31,37,41,43,47,51,53,57,61,67,71,73,79,83,89,97 };

// ---- generate ----

ReceiverSelector* ReceiverSelectorFactory::generate(
	const std::vector<std::string>& param, const size_t nWorker)
{
	ReceiverSelector* res = nullptr;
	const string& name = param[0];
	if(name == "all"){
		res = new ReceiverSelectorBroadcast();
	} else if(name == "ring"){
		res = new ReceiverSelectorRing();
	} else if(name == "random"){
		res = new ReceiverSelectorRandom();
	} else if(name == "hash"){
		res = new ReceiverSelectorHash();
	}
	if(res != nullptr)
		res->init(param, nWorker);
	return res;
}
