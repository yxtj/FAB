#include "Adam.h"
#include <cmath>

using namespace std;

Adam::Adam()
	: lrate(1.0), beta(0.9), beta2(0.999), epsilon(1e-8), t(0)
{
}

void Adam::init(const std::vector<std::string>& param)
{
	if(param.size() > 1)
		lrate = stod(param[1]);
	if(param.size() > 1)
		beta1 = stod(param[2]);
	if(param.size() > 1)
		beta2 = stod(param[3]);
	if(param.size() > 1)
		epsilon = stod(param[4]);
}

void Adam::applyDelta(const std::vector<double>& delta, const double factor)
{
	update_state(delta);
	double fm = 1.0 / (1 - pow(beta1, static_cast<double>(t));
	double fv = 1.0 / (1 - pow(beta2, static_cast<double>(t));
	vector<double> d(delta.size());
	for(size_t i = 0; i < delta.size(); ++i){
		d[i] = (m[i] * fm) / (sqrt(v[i] * fv) + epsilon);
	}
	pm->accumulateParameter(d, lrate*factor);
	++t;
}

void Adam::update_state(const std::vector<double>& grad)
{
	size_t n = grad.size();
	if(t == 0){
		m.assign(n, 0.0);
		v.assign(m, 0.0);
	}
	for(size_t i = 0; i < n; ++i){
		m[i] = beta1 * m[i] + (1 - beta1)*grad[i];
		v[i] = beta2 * v[i] + (1 - beta2)*grad[i] * grad[i];
	}
}
