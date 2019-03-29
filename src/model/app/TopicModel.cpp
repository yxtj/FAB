#include "TopicModel.h"

using namespace std;

void TopicModel::init(const std::string & param)
{
	initBasic(param);
}

bool TopicModel::checkData(const size_t nx, const size_t ny)
{
	return true;
}

std::string TopicModel::name() const
{
	return std::string("tm");
}

bool TopicModel::dataNeedConstant() const
{
	return false;
}

int TopicModel::lengthParameter() const
{
	return nWeight;
}

std::vector<double> TopicModel::predict(const std::vector<double>& x, const std::vector<double>& w) const
{
	return std::vector<double>();
}

int TopicModel::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double TopicModel::loss(const std::vector<double>& pred, const std::vector<double>& label) const
{
	return 0.0;
}

std::vector<double> TopicModel::gradient(const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const
{
	return std::vector<double>();
}
