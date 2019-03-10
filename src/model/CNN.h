#pragma once
#include "Kernel.h"
#include "impl/VectorNetwork.h"

class CNN
	: public Kernel
{
	mutable VectorNetwork net;
public:
	void init(const int xlength, const std::string& param);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	static std::vector<double> gradLoss(const std::vector<double>& pred, const std::vector<double>& label);

	std::vector<double> gradient(
		const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const;
private:
	// make the cnn param into the general format for network
	std::string preprocessParam(const std::string& param);
	// i.e. 5c4p3 -> 5:c:4,sigmoid,max:3
	// i.e. 5c4r,min3 -> 5:c:4,relu,min:3
	std::string procUnitCxPx(const std::string& param);
	// i.e. 5cp2*2 -> 4:c:2*2,sigmoid,max:2*2
	// i.e. 5crp2*2 -> 4:c:2*2,relu,max:2*2
	std::string procUnitCPx(const std::string& param);
};
