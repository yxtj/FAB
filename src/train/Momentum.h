#pragma once
#include "Trainer.h"
#include <string>
#include <vector>

class Momentum : public Trainer
{
	double lrate;
	double alpha;

	std::vector<double> dw;
public:
	// dW = alpha*dW - lrate*grad
	// W = W + dW
	Momentum();
	// momentum:lrate:alpha
	void init(const std::vector<std::string>& param);

	virtual std::vector<double> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true) const;
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true) const;

	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);
private:
	void update_dw(const std::vector<double>& grad);
};

