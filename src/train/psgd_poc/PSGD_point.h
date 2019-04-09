#pragma once
#include "PSGD_poc.h"
#include <vector>
#include <utility>

class PSGD_point : public PSGD_poc
{
	std::vector<float> priority;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void ready();

	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
private:
	std::vector<int> getTopK(const size_t first, const size_t last, const size_t k);
};
