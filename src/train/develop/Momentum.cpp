#include "Momentum.h"

using namespace std;

Momentum::Momentum()
	: lrate(1.0), alpha(0.5)
{
}

void Momentum::init(const std::vector<std::string>& param)
{
	if(param.size() > 1)
		lrate = stod(param[1]);
	if(param.size() > 2)
		alpha = stod(param[2]);
}

void Momentum::applyDelta(const std::vector<double>& delta, const double factor)
{
	update_dw(delta);
	pm->accumulateParameter(dw, factor);
}

void Momentum::update_dw(const std::vector<double>& grad)
{
	size_t n = grad.size();
	if(dw.empty())
		dw.assign(n, 0.0);
	for(size_t i = 0; i < n; ++i)
		dw[i] = alpha * dw[i] - lrate * grad[i];
}
