#pragma once
#include "Trainer.h"

class BlockPSGD : public Trainer
{
	double rate = 1.0;
	double topRatio = 1.0;
	size_t dpblock, n_dpblock; // k-data-point batch (for priority)
	//size_t dmblock, n_dmblock; // k-dimension batch (for priority)
	double renewRatio= 0.01;
	bool useRenewGrad = false;

	size_t paramWidth; // parameter width
	std::vector<std::vector<double>> gradient; // avg gradient of each block
	std::vector<double> sumGrad; // sum of all blocks (NOT all points)
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
	virtual ~BlockPSGD();

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);
private:
	size_t id2block(size_t i) const;
	float calcPriority(const std::vector<double>& g);

	std::vector<int> getTopK(const size_t first, const size_t last, const size_t k);
	void updateGrad(const size_t bid, const std::vector<double>& g, const int cnt);
};
