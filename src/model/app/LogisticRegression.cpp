#include "LogisticRegression.h"
#include "math/activation_func.h"
#include <cmath>
#include <stdexcept>
using namespace std;

void LogisticRegression::init(const std::string & param)
{
	initBasic(param);
	try{
		xlength = stoi(param);
	} catch(...){
		throw invalid_argument("LR parameter is not invalid");
	}
}

bool LogisticRegression::checkData(const size_t nx, const size_t ny)
{
	// check input layer size
	if(nx != stoi(param))
		throw invalid_argument("The dataset does not match the input layer of the network");
	// check output layer size
	if(ny != 1)
		throw invalid_argument("The dataset does not match the output layer of the network");
	return true;
}

std::string LogisticRegression::name() const{
	return "lr";
}

int LogisticRegression::lengthParameter() const
{
	return xlength + 1;
}

std::vector<double> LogisticRegression::predict(
	const std::vector<double>& x, const std::vector<double>& w) const 
{
	double t = w.back();
	for (int i = 0; i < xlength; ++i) {
		t += x[i] * w[i];
	}
	return { sigmoid(t) };
}

int LogisticRegression::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

constexpr double MAX_LOSS = 100;

double LogisticRegression::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	//double cost1 = label * log(pred);
	//double cost2 = (1 - label)*log(1 - pred);
	//return -(cost1 + cost2);
	// the above got overflow
	double cost;
	if(label[0] == 0.0){
		cost = log(1 - pred[0]);
	} else{
		cost = log(pred[0]);
	}
//	if(std::isnan(cost) || std::isinf(cost)) // this std:: is needed for a know g++ bug
	if(std::isinf(cost))
		return MAX_LOSS;
	else
		return -cost;
}

std::vector<double> LogisticRegression::gradient(const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const
{
	// s'(x) = s(x)*(1-s(x))
	// c'(x) = x*(s(x) - y)
	//assert(w.size() == xlength+1);
	double pred = predict(x, w)[0];
	double g0 = pred - y[0];
	vector<double> grad(w.size(), g0);
	for (size_t i = 0; i < xlength; ++i) {
		grad[i] *= x[i];
	}
	return grad;
}

