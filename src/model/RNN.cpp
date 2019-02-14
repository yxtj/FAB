#include "RNN.h"

using namespace std;

void RNN::init(const int xlength, const std::string & param)
{
}

std::string RNN::name() const
{
	return std::string("rnn");
}

bool RNN::dataNeedConstant() const
{
	return false;
}

int RNN::lengthParameter() const
{
	return nWeight;
}

std::vector<double> RNN::predict(const std::vector<double>& x, const std::vector<double>& w) const
{
	return std::vector<double>();
}

int RNN::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double RNN::loss(const std::vector<double>& pred, const std::vector<double>& label) const
{
	return 0.0;
}

std::vector<double> RNN::gradient(const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const
{
	return std::vector<double>();
}
