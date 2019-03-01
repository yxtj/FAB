#pragma once
#include <string>
#include <vector>

struct ReceiverSelector {
public:
	virtual void init(const std::vector<std::string>& param, const size_t nWorker);
	virtual std::vector<int> getTargets(const int sourceId) = 0;
protected:
	int nw;
};

struct ReceiverSelectorFactory {
	// param[0] is the name of strategy
	// Support: "all" -> broadcast to every worker
	//          "ring" -> send to the next k workers (param: k [=2])
	//          "random" -> send to random k workers (param: k, seed)
	//          "hash" -> send to fixed k workers (param: k)
	static ReceiverSelector* generate(const std::vector<std::string>& param, const size_t nWorker);
};
