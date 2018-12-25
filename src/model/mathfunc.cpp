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
