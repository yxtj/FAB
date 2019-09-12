#include "accumulate.h"
#include <algorithm>
#include <numeric>

using namespace std;

double sum(const std::vector<double>& vec){
	return accumulate(vec.begin(), vec.end(), 0.0);
}

double mean(const std::vector<double>& vec){
	if(vec.empty())
		return 0.0;
	return accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

double hmean(const std::vector<double>& vec){
	if(vec.empty())
		return 0.0;
	double res = 0.0;
	for(auto& v : vec){
		res += 1.0 / v;
	}
	return vec.size() / res;
}

double minimum(const std::vector<double>& vec, const double init){
	auto it = min_element(vec.begin(), vec.end());
	if(it == vec.end())
		return init;
	return *it;
}

double maximum(const std::vector<double>& vec, const double init){
	auto it = max_element(vec.begin(), vec.end());
	if(it == vec.end())
		return init;
	return *it;
}
