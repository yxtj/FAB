#pragma once
#include "model/Model.h"
#include "data/DataHolder.h"
#include <utility>
#include <vector>
#include <atomic>

class Trainer
{
public:
	Model* pm;
	const DataHolder* pd;
public:
	virtual void init(const std::vector<std::string>& param) = 0;
	virtual std::string name() const = 0;
	std::vector<std::string> getParam() const;

	void bindModel(Model* pm);
	void bindDataset(const DataHolder* pd);
	// last step before running
	virtual void ready();

	double loss(const size_t topn = 0) const;

	// calculate the delta values to update the model parameter
	// <cnt> = 0 means use all the data points.
	// <avg> is set to true by default. Note that it may not be used for some models
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true) = 0;
	// try to use all data points in given range, unless the condition is set to false before finish.
	// <cond> is the continue condition, it can be changed in another thread.
	// return the number of used data points.
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true) = 0;

	// apply the delta values to the model parameter, parameter += delat*factor
	virtual void applyDelta(const std::vector<double>& delta, const double factor = 1.0);

protected:
	std::vector<std::string> param;
	void initBasic(const std::vector<std::string>& param);
};
