#pragma once
#include "Trainer.h"

class PSGD : public Trainer
{
	double rate = 1.0;
	double topRatio = 1.0;
	bool global = true; // true->global (pi=gi*avg(g)), false->self (pi=gi*gi)
	size_t paramWidth; // parameter width
	std::vector<std::vector<double>> gradient;
	std::vector<double> sumGrad;
	std::vector<float> priority;

public:
	double stat_t_grad_calc = 0, stat_t_grad_post = 0;
	double stat_t_prio_pick = 0, stat_t_prio_update = 0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void prepare();
	virtual void ready();
	virtual ~PSGD();

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);
private:
	float calcPriority(const std::vector<double>& g);
	float calcPriorityGlobal(const std::vector<double>& g);
	float calcPrioritySelf(const std::vector<double>& g);

	using fp_cp_t = decltype(&PSGD::calcPriority);
	fp_cp_t fp_cp;

	std::vector<int> getTopK(const size_t first, const size_t last, const size_t k);
	void updateSumGrad(const int i, const std::vector<double>& g);
};
