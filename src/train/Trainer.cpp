#include "Trainer.h"
using namespace std;

void Trainer::bindModel(Model* pm){
	this->pm = pm;
}

void Trainer::bindDataset(const DataHolder* pd){
	this->pd = pd;
}

void Trainer::prepare()
{
}

void Trainer::ready()
{
}

void Trainer::initBasic(const std::vector<std::string>& param)
{
	this->param = param;
}

std::vector<std::string> Trainer::getParam() const
{
	return param;
}

bool Trainer::needAveragedDelta() const
{
	return true;
}

double Trainer::loss(const size_t topn) const {
	double res = 0;
	size_t n = topn == 0 ? pd->size() : topn;
	for(size_t i = 0; i < n; ++i){
		res += pm->loss(pd->get(i));
	}
	return res / static_cast<double>(n);
}

void Trainer::applyDelta(const vector<double>& delta, const double factor)
{
	pm->accumulateParameter(delta, factor);
}
