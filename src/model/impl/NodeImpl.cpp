#include "NodeImpl.h"
#include "math/activation_func.h"
#include <cassert>
#include <algorithm>
//#include <numeric>

using namespace std;

// ---- Input Node ----

InputNode::InputNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{}

std::vector<double> InputNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	return x;
}

std::vector<double> InputNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	return std::vector<double>();
}


// ---- Convolutional Node: 1D ----

ConvNode1D::ConvNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), k(shape[0])
{
	assert(shape.size() == 1);
	assert(k > 0);
}

std::vector<double> ConvNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t ny = x.size() - k + 1;
	std::vector<double> res(ny);
	for (size_t i = 0; i < ny; ++i) {
		double t = 0.0;
		for (size_t j = 0; j < k; ++j)
			t += x[i + j] * w[off + j];
		res[i] = t;
	}
	return res;
}

std::vector<double> ConvNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size() + nw - 1);
	assert(y.size() == pre.size());
	const size_t nx = x.size();
	const size_t ny = y.size();
	// dy/dw
	for (size_t i = 0; i < nw; ++i) {
		double t = 0.0;
		for (size_t j = 0; j < ny; ++j) {
			t += pre[j] * x[j + i];
		}
		grad[off + i] += t;
	}
	// dy/dx
	std::vector<double> res(nx);
	for (size_t i = 0; i < nx; ++i) {
		double t = 0.0;
		// cut the first and the last
		for (size_t j = (i < ny ? 0 : i - ny + 1); j < nw && i >= j; ++j) {
			t += pre[i - j] * w[off + j];
		}
		res[i] = t;
	}
	return res;
}

// ---- Recurrent Node ----

RecuNode::RecuNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), k(shape[1])
{
	last_pred.assign(n, 0.0);
	last_grad.assign(n, 0.0);
}

std::vector<double> RecuNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	std::vector<double> res(k);
	// element by element (n + k + 1)
	size_t p = off;
	for(int i = 0; i < k; ++i){
		double a = 0.0;
		for(int j = 0; j < n; ++j){ // W*x
			a += x[j] * w[p++];
		}
		double b = 0.0;
		for(int j = 0; j < k; ++j){ // U*y
			b += last_pred[j] * w[p++];
		}
		res[i] = a + b; // + w[p++];
	}
	// store current output for next call
	last_pred = res;
	return res;
}

std::vector<double> RecuNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == n);
	assert(y.size() == k);
	const size_t n = y.size();
	std::vector<double> res(n); // dy/dx
	const size_t ngroup = n + k;// +1;
	size_t p = off;
	for(int i = 0; i < k; ++i){
		double f = pre[i];
		for(int j = 0; j < n; ++j){ // W*x
			grad[p] += f * x[j]; // dy/dw
			res[j] += f * w[p]; // dy/dx
			++p;
		}
		for(int j = 0; j < k; ++j){ // U*y
			grad[p++] += f * last_grad[j]; // dy/dw
		}
		//grad[p++] = 1.0; // dy/dw
	}
	// store current output for next call
	last_grad = y;
	return res;
}

// ---- Activation Node: ReLU ----

ReluNode::ReluNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 1;
}

std::vector<double> ReluNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = x.size();
	std::vector<double> res(n);
	double t = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res[i] = relu(x[i] + w[off]);
	}
	return res;
}

std::vector<double> ReluNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size());
	const size_t n = y.size();
	std::vector<double> res(n); // dy/dx
	double s = 0.0;
	for (size_t i = 0; i < n; ++i) {
		double d = relu_derivative(x[i] + w[off]);
		double f = pre[i] * d;
		s += f * x[i]; // dy/dw
		res[i] = f * w[off]; // dy/dx
	}
	grad[off] += s / n;
	return res;
}

// ---- Activation Node: Sigmoid ----

SigmoidNode::SigmoidNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 1;
}

std::vector<double> SigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = x.size();
	std::vector<double> res(n);
	for (size_t i = 0; i < n; ++i) {
		res[i] = sigmoid(x[i] + w[off]);
	}
	return res;
}

std::vector<double> SigmoidNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size());
	const size_t n = y.size();
	std::vector<double> res(n);
	double s = 0.0;
	for (size_t i = 0; i < n; ++i) {
		//double d = sigmoid_derivative(x[i] + w[off], y[i]);
		double d = sigmoid_derivative(0.0, y[i]);
		double f = pre[i] * d;
		s += f * x[i]; // dy/dw
		res[i] = f * w[off]; // dy/dx
	}
	grad[off] += s / n;
	return res;
}

// ---- Pooling Node: 1D max ----

PoolMaxNode1D::PoolMaxNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
	assert(shape.size() == 1);
}

std::vector<double> PoolMaxNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t step = shape[0];
	const size_t n = (x.size() + step - 1) / step;
	vector<double> res(n);
	for (size_t i = 0; i < n; ++i) {
		double v = x[i*step];
		size_t limit = min((i + 1)*step, x.size());
		for (size_t j = i * step + 1; j < limit; ++j)
			v = max(v, x[j]);
		// TODO: store the max index for gradient
		res[i] = v;
	}
	return res;
}

std::vector<double> PoolMaxNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// no weight -> no change on <grad>
	// if argmax(x[1],...,x[n]) = i , then dy/dx = 1.0 and 0 for others
	const size_t step = shape[0];
	const size_t ny = y.size();
	vector<double> res(x.size(), 0.0);
	for (size_t i = 0; i < ny; ++i) {
		size_t limit = min((i + 1)*step, x.size());
		for (size_t j = i * step; j < limit; ++j) {
			if (x[j] == y[i])
				res[j] = pre[i];
		}
	}
	return res;
}

// ---- Fully-Connected Node: 1D ----

FCNode1D::FCNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw += 1;
}

double FCNode1D::predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	double res = 0.0;
	size_t p = off;
	for (size_t i = 0; i < n1; ++i) {
		for (size_t j = 0; j < n2; ++j)
			res += x[i][j] * w[p++];
	}
	return sigmoid(res + w[p]);
}

std::vector<std::vector<double>> FCNode1D::gradient(
	std::vector<double>& grad, const std::vector<std::vector<double>>& x,
	const std::vector<double>& w, const double& y, const double& pre)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	//assert(y.size() == 1 && pre.size() == 1);
	//const double f = pre[0] * y[0];
	const double d = sigmoid_derivative(0.0, y);
	const double f = pre * d;
	std::vector<std::vector<double>> pg(n1, vector<double>(n2));
	size_t p = off;
	for (size_t i = 0; i < n1; ++i) {
		for (size_t j = 0; j < n2; ++j) {
			grad[p] = x[i][j] * f; // pre * dy/dw
			pg[i][j] = w[p] * f; // pre * dy/dx
			++p;
		}
	}
	grad[p] += f; // the constant offset
	return pg;
}
