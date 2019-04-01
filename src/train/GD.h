#pragma once
#include "Trainer.h"

class GD : public Trainer
{
	double rate = 1.0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	void setRate(const double rate);
	double getRate() const;
	virtual void ready();

	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);

};

