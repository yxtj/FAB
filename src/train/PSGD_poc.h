#pragma once
#include "Trainer.h"

class PSGD_poc : public Trainer
{
	double rate = 1.0;
	bool mergeDim;
	size_t paramWidth; // parameter width

public:
	double stat_t_priority = 0, stat_t_grad = 0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void ready();
	virtual ~PSGD_poc();

	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
private:
	std::pair<size_t, std::vector<double>> batchDelta_point(
		const size_t start, const size_t cnt, const bool avg);
	std::pair<size_t, std::vector<double>> batchDelta_dim(
		const size_t start, const size_t cnt, const bool avg);
};
