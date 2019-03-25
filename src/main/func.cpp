#include "func.h"
#include <cmath>

using namespace std;

double vectorDifference(const vector<double>& a, const vector<double>& b){
	double res = 0.0;
	size_t n = a.size();
	for(size_t i = 0; i < n; ++i){
		double t = a[i] - b[i];
		res += t * t;
	}
	return sqrt(res);
}
