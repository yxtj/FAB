#pragma once
#include "model/Kernel.h"
#include "model/impl/MLPProxy.h"

class MLP
	: public Kernel
{
	int nLayer;
	std::vector<int> nNodeLayer;
	mutable MLPProxy proxy; // non-const bind function required
public:
	void init(const std::string& param);
	bool checkData(const size_t nx, const size_t ny);
	std::string name() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w) const;
	int classify(const double p) const;
	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;

	std::vector<double> forward(const std::vector<std::vector<double>>& x, const std::vector<double>& w);
	std::vector<double> backward(const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph = nullptr);
	std::vector<double> gradient(const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph = nullptr) const;
private:
	double getWeight(const std::vector<double>& w, const int layer, const int from, const int to) const;

	// require: proxy.bind(&w) had been called before
	std::vector<double> activateLayer(
		const std::vector<double>& x, const std::vector<double>& w, const int layer) const;
private:
	std::vector<std::vector<double>> mid;
};
