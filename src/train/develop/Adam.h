#pragma once
#include "Trainer.h"
#include <string>
#include <vector>

class Adam : public Trainer
{
	double lrate;
	double beta1, beta2;
	double epsilon

	std::vector<double> m, v;
	int t;
public:
	// m_t = beta_1 * m_t + (1-beta_1) * grad
	// v_t = beta_2 * b_t + (1-beta_2) * grad^2
	// m_t' = m_t / (1-beta_1^t)
	// v_t' = v_t / (1-beta_2^t)
	// W = w - lrate / (sqrt(v_t') + epsilon) * m_t'
	Adam();
	// momentum:lrate:alpha
	void init(const std::vector<std::string>& param);

	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);
private:
	void update_state(const std::vector<double>& grad);
};

