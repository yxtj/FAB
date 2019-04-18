#pragma once
#include "model/Kernel.h"
#include "model/impl/VectorNetwork.h"

class RNN
	: public Kernel
{
	mutable VectorNetwork net;
public:
	void init(const std::string& param);
	bool checkData(const size_t nx, const size_t ny);
	std::string name() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	static std::vector<double> gradLoss(const std::vector<double>& pred, const std::vector<double>& label);

	std::vector<double> gradient(const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph = nullptr) const;

	// make param into general format for network
	std::string preprocessParam(const std::string& param);
private:
};
