#pragma once
#include "model/Kernel.h"
#include <vector>

class LogisticRegression
	: public Kernel
{
public:
	void init(const std::string& param);
	bool checkData(const size_t nx, const size_t ny);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	std::vector<double> gradient(
		const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const;

private:
	int xlength;
};
