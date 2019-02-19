#include "norm.h"

using namespace std;

double l1norm(const std::vector<double>& vec, const bool avg) {
	double r = 0.0;
	for (auto& v : vec)
		r += (v >= 0.0 ? v : -v);
	if(avg)
		r /= vec.size();
	return r;
}

double l2norm(const std::vector<double>& vec, const bool avg) {
	double r = 0.0;
	for (auto& v : vec)
		r += v * v;
	if(avg)
		r /= vec.size();
	return r;
}

double l1norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2, const bool avg)
{
	double r = 0.0;
	const size_t n = vec1.size();
	for (size_t i = 0; i < n;++i) {
		double v = vec1[i] - vec2[i];
		r += (v >= 0.0 ? v : -v);
	}
	if(avg)
		r /= n;
	return r;
}

double l2norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2, const bool avg)
{
	double r = 0.0;
	const size_t n = vec1.size();
	for (size_t i = 0; i < n; ++i) {
		double v = vec1[i] - vec2[i];
		r += v * v;
	}
	if(avg)
		r /= n;
	return r;
}
