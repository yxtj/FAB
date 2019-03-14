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

	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);
private:
	void update_dw(const std::vector<double>& grad);
};

