#pragma once
#include "Trainer.h"

class PSGDDecay : public Trainer
{
	double rate = 1.0;
	double topRatio = 1.0;
	double decay = 0.1;
	double renewRatio = 0.01;
	bool useRenewGrad = false;

	size_t paramWidth; // parameter width
	std::vector<double> avgGrad;
	std::vector<float> priority;
	size_t renewSize;
	size_t renewPointer;

public:
	double stat_t_grad_renew = 0, stat_t_grad_calc = 0, stat_t_grad_post = 0;
	double stat_t_prio_pick = 0, stat_t_prio_update = 0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void prepare();
	virtual void ready();
	virtual ~PSGDDecay();

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);
private:
	float calcPriority(const std::vector<double>& g);

	std::vector<int> getTopK(const size_t first, const size_t last, const size_t k);
	void updateAvgGrad(const std::vector<double>& g);
	void updateAvgGrad(const std::vector<double>& g, const double f);
};
