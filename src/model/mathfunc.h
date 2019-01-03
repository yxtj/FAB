#pragma once
#include <vector>

double sigmoid(double x);
double sigmoid_derivative(double y);

double relu(double x);
double relu_derivative(double y);

//double tanh(double x);
double tanh_derivative(double y);

double arctan(double x);
double arctan_derivative(double y);

double softplus(double x);
double softplus_derivative(double y);


std::vector<double> softmax(const std::vector<double>& y);
