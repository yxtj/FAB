#pragma once
#include "Trainer.h"

// hidden variable + non-average
class EM_KMeans : public Trainer
{
	std::vector<std::vector<double>> h; // hidden variable

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual bool needAveragedDelta() const;
	virtual void prepare();

	// special proecss on the <n> part of weight
	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg, const double adjust);

};

