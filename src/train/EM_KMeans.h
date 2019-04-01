#pragma once
#include "Trainer.h"

class EM_KMeans : public Trainer
{
	std::vector<std::vector<double>> h; // hidden variable

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual bool needAveragedDelta() const;
	virtual void ready();

	// special proecss on the <n> part of weight
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);

};

