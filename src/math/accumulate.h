#include <vector>
#include <limits>

double sum(const std::vector<double>& vec);
double mean(const std::vector<double>& vec);
double hmean(const std::vector<double>& vec); // harmonic mean

#if defined(_WIN32) || defined(_WIN64)
#undef max
#endif // windows

double minimum(const std::vector<double>& vec, const double init = std::numeric_limits<double>::max());
double maximum(const std::vector<double>& vec, const double init = std::numeric_limits<double>::lowest());
