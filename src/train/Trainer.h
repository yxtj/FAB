#pragma once
#include "model/Model.h"
#include "data/DataHolder.h"
#include <utility>
#include <vector>
#include <atomic>

class Trainer
{
public:
	Model* pm =nullptr;
	const DataHolder* pd = nullptr;

	struct DeltaResult {
		size_t n_scanned;
		size_t n_reported;
		std::vector<double> delta;
		double loss;
	};
public:
	virtual void init(const std::vector<std::string>& param) = 0;
	virtual std::string name() const = 0;
	std::vector<std::string> getParam() const;
	virtual bool needAveragedDelta() const;

	void bindModel(Model* pm);
	void bindDataset(const DataHolder* pd);
	// called after bind model and dataset (without parameter)
	virtual void prepare();
	// last step before running
	virtual void ready();
	virtual ~Trainer(){}

	double loss(const size_t topn = 0) const;

	// calculate the delta values to update the model parameter
	// if <cond> is reset, finish as soon as possible
	// <cnt> = 0 means use all the data points.
	// <avg> is set to true by default. Note that it may not be used for some models
	// <cond> is the continue condition, it can be changed in another thread.
	// return the number of used data points.
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true) = 0;
	// slowdown the processing by <slow> of the normal processing time
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg, const double slow);

	// apply the delta values to the model parameter, parameter += delat*factor
	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);

protected:
	std::vector<std::string> param;
	void initBasic(const std::vector<std::string>& param);
};
