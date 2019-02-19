#pragma once
#include <vector>

double l1norm(const std::vector<double>& vec, const bool avg = false);
double l2norm(const std::vector<double>& vec, const bool avg = false);

double l1norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2, const bool avg = false);
double l2norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2, const bool avg = false);
