#pragma once
#include "model/Model.h"
#include "data/DataHolder.h"
#include <utility>
#include <atomic>

class Trainer
{
public:
	Model* pm;
	const DataHolder* pd;
public:
	void bindModel(Model* pm);
	void bindDataset(const DataHolder* pd);

	double loss(const size_t topn = 0) const;

	// train using all the data in given range.
	// <cnt> = 0 means use all the data points.
	virtual void train(const size_t start = 0, const size_t cnt = 0);
	// try to use all data points in given range, unless the condition is set to false before finish.
	// <cond> is the continue condition, it can be changed in another thread.
	// return the number of used data points.
	virtual size_t train(std::atomic<bool>& cond, const size_t start = 0, const size_t cnt = 0);

	// calculate the delta values to update the model parameter
	// <avg> is set to true by default. Note that it may not be used for some models
	virtual std::vector<double> batchDelta(const size_t start, const size_t cnt, const bool avg = true) const = 0;
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true) const = 0;

	// apply the delta values to the model parameter, parameter += delat*factor
	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);
};

