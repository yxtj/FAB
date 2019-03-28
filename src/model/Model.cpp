#include "Model.h"
#include "KernelFactory.h"

using namespace std;

/*
void Model::initParamWithData(const DataPoint& d){
	param.init(d.x.size(), 0.01);
}

void Model::initParamWithSize(const size_t n){
	param.init(n, 0.01);
}*/

void Model::init(const std::string& name, const std::string & paramKern)
{
	generateKernel(name);
	kern->init(paramKern);
}

void Model::init(const std::string& name, const std::string & paramKern, const double w0)
{
	generateKernel(name);
	kern->init(paramKern);
	size_t n = kern->lengthParameter();
	param.init(n, w0);
}

void Model::init(const std::string & name, const std::string & paramKern, const unsigned seed)
{
	generateKernel(name);
	kern->init(paramKern);
	size_t n = kern->lengthParameter();
	param.init(n, 0.01, 0.01, seed);
}

bool Model::checkData(const size_t nx, const size_t ny)
{
	return kern->checkData(nx, ny);
}

void Model::clear()
{
	delete kern;
	kern = nullptr;
}

std::string Model::kernelName() const{
	return kern->name();
}

void Model::setParameter(const Parameter& p) {
	param = p;
}
void Model::setParameter(Parameter&& p) {
	param = move(p);
}

Parameter & Model::getParameter()
{
	return param;
}

const Parameter & Model::getParameter() const
{
	return param;
}

size_t Model::paramWidth() const
{
	return kern->lengthParameter();
}

Kernel* Model::getKernel(){
	return kern;
}

void Model::accumulateParameter(const std::vector<double>& grad, const double factor)
{
	param.accumulate(grad, factor);
}

void Model::accumulateParameter(const std::vector<double>& grad)
{
	param.accumulate(grad);
}

std::vector<double> Model::predict(const DataPoint& dp) const
{
	return kern->predict(dp.x, param.weights);
}

int Model::classify(const double p) const
{
	return kern->classify(p);
}

double Model::loss(const DataPoint & dp) const
{
	std::vector<double> pred = kern->predict(dp.x, param.weights);
	return loss(pred, dp.y);
}

double Model::loss(const std::vector<double>& pred, const std::vector<double>& label) const
{
	return kern->loss(pred, label);
}

std::vector<double> Model::gradient(const DataPoint & dp) const
{
	return kern->gradient(dp.x, param.weights, dp.y);
}

void Model::generateKernel(const std::string & name)
{
	if(kern != nullptr){
		delete kern;
		kern = nullptr;
	}
	kern = KernelFactory::generate(name);
}
