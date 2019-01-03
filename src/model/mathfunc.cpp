#include "mathfunc.h"
#include <cmath>
using namespace std;

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double y)
{
	double t = sigmoid(y);
	return t * (1.0 - t);
}

double relu(double x){
	return x >= 0.0 ? x : 0;
}

double relu_derivative(double x){
	return x >=0.0 ? 1 : 0;
}

double tanh_derivative(double y){
	double t = tanh(y);
	return 1 - t*t;
}

double arctan(double x){
	return atan(x);
}

double arctan_derivative(double y){
	return 1.0/(y*y + 1);
}

double softplus(double x){
	return log1p(exp(x));
}
double softplus_derivative(double y){
	return sigmoid(y);
}

std::vector<double> softmax(const std::vector<double>& x){
	std::vector<double> res(x.size());
	double sum=0.0;
	for(size_t i=0;i<x.size();++i){
		double t = exp(x[i]);
		res[i]=t;
		sum+=t;
	}
	for(double& v : res){
		v/=sum;
	}
	return res;
}

std::vector<std::vector<double>> softmax_derivative(const std::vector<double>& y){
	const size_t n = y.size();
	std::vector<std::vector<double>> res(n, vector<double>(n));
	for(size_t i=0;i<n;++i){
		for(size_t j=0;j<n;++j)
			if(i==j)
				res[i][j]=y[i]*(1-y[j]);
			else
				res[i][j]=-y[i]*y[j];
	}
	return res;
}
