#pragma once
#include "model/Kernel.h"
#include <vector>

class KMeans
	: public Kernel
{
public:
	void init(const std::string& param);
	bool checkData(const size_t nx, const size_t ny);
	std::string name() const;
	int lengthParameter() const;
	bool needInitParameterByData() const;

	int lengthHidden() const;
	void initVariables(const std::vector<std::vector<double>>& x,
		std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph);

	std::vector<double> predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	// ph stores the current assignment of the node
	std::vector<double> gradient(const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const;

private:
	using it_t = typename std::vector<double>::const_iterator;
	static double dist(it_t xf, it_t xl, it_t yf, const double n);
	// sum (x_i - y_i/n)^2 . x is fixed and y changes. 
	// change to sum(y_i^2) - 2*n*sum ( x_i - y_i)^2
	static double quickDist(it_t xf, it_t xl, it_t yf, const double n);
	// size_t quickPredict(const std::vector<double>& x, const std::vector<double>& w) const;
	// return cluster id and obj value
	pair<size_t, double> quickPredict(const std::vector<double>& x, const std::vector<double>& w) const;
private:
	size_t dim;
	size_t ncenter;
	size_t parlen;
};
