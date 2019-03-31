#include "KMeans.h"
#include "math/activation_func.h"
#include "util/Util.h"
#include <cmath>
#include <stdexcept>

using namespace std;

void KMeans::init(const std::string & param)
{
	initBasic(param);
	try{
		auto vec = getIntList(param, ":-, ");
		ncenter = vec[0];
		dim = vec[1];
		parlen = ncenter * (dim + 1);
	} catch(...){
		throw invalid_argument("KMeans parameter is not invalid");
	}
}

bool KMeans::checkData(const size_t nx, const size_t ny)
{
	// check input layer size
	if(nx != dim)
		throw invalid_argument("The dataset does not match the input layer of the network");
	// check output layer size
	if(ny != 0)
		throw invalid_argument("The dataset does not match the output layer of the network");
	return true;
}

std::string KMeans::name() const{
	return "km";
}

bool KMeans::dataNeedConstant() const{
	return false;
}

int KMeans::lengthParameter() const
{
	return parlen;
}

std::vector<double> KMeans::predict(
	const std::vector<double>& x, const std::vector<double>& w) const
{
	size_t min_id = ncenter;
	double min_v;
	size_t off = 0;
	for(size_t i = 0; i < ncenter; ++i){
		double d = dist(x.begin(), x.end(), w.begin() + off, w[off + dim + 1]);
		off += dim + 1;
		if(min_id == ncenter || d < min_v){
			min_id = i;
			min_v = d;
		}
	}
	return { static_cast<double>(min_id), min_v };
}

int KMeans::classify(const double p) const
{
	return 0;
}

double KMeans::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	return pred[1];
}

std::vector<double> KMeans::gradient(const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const
{
	vector<double> grad(parlen, 0.0);
	size_t oldp = static_cast<int>((*ph)[0]);
	size_t newp = quickPredict(x, w);
	if(oldp != newp){
		oldp *= dim + 1;
		newp *= dim + 1;
		for(size_t i = 0; i < dim; ++i){
			grad[oldp + i] -= x[i];
			grad[newp + i] += x[i];
		}
		grad[oldp + dim] -= 1;
		grad[newp + dim] += 1;
	}
	return grad;
}

double KMeans::dist(it_t xf, it_t xl, it_t yf, const double n)
{
	double r = *xf - *yf / n;
	r *= r;
	while(++xf != xl){
		++yf;
		double t = *xf - *yf / n;
		r += t * t;
	}
	return sqrt(r);
}

double KMeans::quickDist(it_t xf, it_t xl, it_t yf, const double n)
{
	double yy = 0.0;
	double xy = 0.0;
	while(xf != xl){
		xy += *xf * *yf;
		yy += *yf * *yf;
		++xf;
		++yf;
	}
	return yy - 2 * n*xy;
}

size_t KMeans::quickPredict(const std::vector<double>& x, const std::vector<double>& w) const
{
	int min_id = -1;
	double min_v;
	size_t off = 0;
	for(size_t i = 0; i < ncenter; ++i){
		double d = quickDist(x.begin(), x.end(), w.begin() + off, w[off + dim + 1]);
		off += dim + 1;
		if(min_id == -1 || d < min_v){
			min_id = i;
			min_v = d;
		}
	}
	return min_id;
}

