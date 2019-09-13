#pragma once
#include "Trainer.h"

class GD : public Trainer
{
	double rate = 1.0;
	double stat_t_grad_calc = 0, stat_t_grad_post= 0;
public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	void setRate(const double rate);
	double getRate() const;
	virtual void prepare();
	virtual ~GD();

	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg, const double adjust);

};

