#pragma once
#include "Trainer.h"
#include "impl/PrioritizedHolder.hpp"
#include <unordered_map>

class BlockPSGD : public Trainer
{
	double rate = 1.0;
	double kratio;
	size_t dpblock, n_dpblock; // k-data-point batch (for priority)
	size_t dmblock, n_dmblock; // k-dimension batch (for priority)
	size_t paramWidth; // parameter width

	struct hash_coord {
		std::hash<int> h;
		size_t operator()(const std::pair<int, int>& p) const{
			return h((p.second << 6) | p.first);
		}
	};

	PrioritizedHolder<std::pair<int, int>, hash_coord> priQue;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void prepare();
	virtual void ready();

	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
private:
	std::vector<float> calcPriority(const std::vector<double>& grad);
	std::unordered_map<int, std::vector<int>> mergeSelected(const std::vector<std::pair<int, int>>& pick);
};
