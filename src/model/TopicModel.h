#pragma once
#include "Kernel.h"

class TopicModel
	: public Kernel
{
	int nLayer;
	std::vector<int> nNodeLayer;
	int nWeight;
public:
	void init(const int xlength, const std::string& param);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	std::vector<double> gradient(
		const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const;
private:

};
