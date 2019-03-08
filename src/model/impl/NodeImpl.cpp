#include "NodeImpl.h"
#include "math/activation_func.h"
#include <cassert>
#include <algorithm>
#include <limits>
//#include <numeric>

using namespace std;

// ---- Input Node ----

InputNode::InputNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
}

std::vector<double> InputNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	return x;
}

std::vector<double> InputNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	return std::vector<double>();
}


// ---- Weighted Summation Node ----

WeightedSumNode::WeightedSumNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), k(shape[1])
{
	assert(shape.size() == 2);
	assert(n > 0);
	assert(k > 0);
	nw += k; // the constant offsets
}

std::vector<int> WeightedSumNode::outShape(const std::vector<int>& inShape) const
{
	return { k };
}

std::vector<double> WeightedSumNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	assert(x.size() == n);
	vector<double> res(k);
	int idx = off;
	for(int i = 0; i < k;  ++i){
		// idx = off + i * n;
		double v = 0.0;
		for(int j = 0; j < n; ++j)
			v += w[idx++] * x[j];
		v += w[idx++];
		res[i] = v;
	}
	return res;
}

std::vector<double> WeightedSumNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == n);
	assert(y.size() == k);
	vector<double> res(n);
	int idx = off;
	for(int i = 0; i < k; i++){
		// idx = off + i * n;
		const double factor = pre[i];
		for(int j = 0; j < n; ++j){
			grad[idx] += factor * x[j]; // dy/dw
			res[j] += factor * w[idx]; // dy/dx
			++idx;
		}
		grad[idx++] += factor * 1; // dy/dw
	}
	return res;
}

// ---- Convolutional Node: 1D ----

ConvNode1D::ConvNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), k(shape[0])
{
	assert(shape.size() == 1);
	assert(k > 0);
	nw += 1; // the constant offset
}

std::vector<int> ConvNode1D::outShape(const std::vector<int>& inShape) const
{
	return { inShape[0] - k + 1 };
}

std::vector<double> ConvNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t ny = x.size() - k + 1;
	std::vector<double> res(ny);
	for (size_t i = 0; i < ny; ++i) {
		double t = w[off + k];
		for (size_t j = 0; j < k; ++j)
			t += x[i + j] * w[off + j];
		res[i] = t;
	}
	return res;
}

std::vector<double> ConvNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size() + k - 1);
	assert(y.size() == pre.size());
	const size_t nx = x.size();
	const size_t ny = y.size();
	// dy/dw
	for (size_t i = 0; i < k; ++i) {
		double t = 0.0;
		for (size_t j = 0; j < ny; ++j) {
			t += pre[j] * x[i + j];
		}
		grad[off + i] += t;
	}
	for(size_t j = 0; j < ny; ++j) {
		grad[off + k] += pre[j];
	}
	// dy/dx
	std::vector<double> res(nx);
	for (size_t i = 0; i < nx; ++i) {
		double t = 0.0;
		// cut the first (i >= j)
		// cut the last (i < ny ? 0 : i - ny + 1)
		for (size_t j = (i < ny ? 0 : i - ny + 1); j < k && i >= j; ++j) {
			t += pre[i - j] * w[off + j];
		}
		res[i] = t;
	}
	return res;
}

// ---- Convolutional Node: 2D ----

ConvNode2D::ConvNode2D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), m(shape[1]), k1(shape[2]), k2(shape[3])
{
	assert(shape.size() == 4);
	assert(k1 > 0 && k2 > 0);
	nw += 1; // the constant offset
}

std::vector<int> ConvNode2D::outShape(const std::vector<int>& inShape) const
{
	return { inShape[0] - k1 + 1, inShape[1] - k2 + 1 };
}

std::vector<double> ConvNode2D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	// TODO:
	return std::vector<double>();
}

std::vector<double> ConvNode2D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// TODO:
	return std::vector<double>();
}

// ---- Recurrent Node Base ----

RecurrentNodeBase::RecurrentNodeBase(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), k(shape[1])
{
	last_pred.assign(n, 0.0);
	last_grad.assign(n, 0.0);
	nw = (n + k + 1)*k;
}

std::vector<int> RecurrentNodeBase::outShape(const std::vector<int>& inShape) const
{
	return { k };
}

std::vector<double> RecurrentNodeBase::predictCalcOnly(const std::vector<double>& x, const std::vector<double>& w)
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
		res[i] = a + b + w[p++];
	}
	return res;
}

std::vector<double> RecurrentNodeBase::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	auto res = predict(x, w);
	last_pred = res;
	return res;
}

std::vector<double> RecurrentNodeBase::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == n);
	assert(y.size() == k);
	std::vector<double> res(n); // dy/dx
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
		grad[p++] += f; // b, dy/dw
	}
	// store current output for next call
	last_grad = y;
	return res;
}

// ---- Recurrent Node Sigmoid ----

RecurrentSigmoidNode::RecurrentSigmoidNode(const size_t offset, const std::vector<int>& shape)
	: RecurrentNodeBase(offset, shape)
{}

std::vector<double> RecurrentSigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	auto res = predictCalcOnly(x, w);
	for(auto& v : res)
		v = sigmoid(v);
	last_pred = res;
	return res;
}

std::vector<double> RecurrentSigmoidNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	auto actPre = pre;
	for(size_t i=0;i<pre.size();++i)
		actPre[i] *= sigmoid_derivative(0.0, y[i]);
	return RecurrentNodeBase::gradient(grad, x, w, y, actPre);
}

// ---- Recurrent Node Tanh ----

RecurrentTanhNode::RecurrentTanhNode(const size_t offset, const std::vector<int>& shape)
	: RecurrentNodeBase(offset, shape)
{}

std::vector<double> RecurrentTanhNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	auto res = predictCalcOnly(x, w);
	for(auto& v : res)
		v = tanh(v);
	last_pred = res;
	return res;
}

std::vector<double> RecurrentTanhNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	auto actPre = pre;
	for(size_t i = 0; i < pre.size(); ++i)
		actPre[i] *= tanh_derivative(0.0, y[i]);
	return RecurrentNodeBase::gradient(grad, x, w, y, actPre);
}

// ---- Activation Node: ReLU ----

ReluNode::ReluNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
}

std::vector<double> ReluNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = x.size();
	std::vector<double> res(n);
	for (size_t i = 0; i < n; ++i) {
		res[i] = relu(x[i]);
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
		double d = relu_derivative(x[i]);
		double f = pre[i] * d;
		res[i] = f; // dy/dx
	}
	return res;
}

// ---- Activation Node: Sigmoid ----

SigmoidNode::SigmoidNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
}

std::vector<double> SigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = x.size();
	std::vector<double> res(n);
	for (size_t i = 0; i < n; ++i) {
		res[i] = sigmoid(x[i]);
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
		//double d = sigmoid_derivative(x[i], y[i]);
		double d = sigmoid_derivative(0.0, y[i]);
		double f = pre[i] * d;
		res[i] = f; // dy/dx
	}
	return res;
}

// ---- Activation Node: Tanh ----

TanhNode::TanhNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
}

std::vector<double> TanhNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = x.size();
	std::vector<double> res(n);
	for(size_t i = 0; i < n; ++i) {
		res[i] = tanh(x[i]);
	}
	return res;
}

std::vector<double> TanhNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size());
	const size_t n = y.size();
	std::vector<double> res(n);
	double s = 0.0;
	for(size_t i = 0; i < n; ++i) {
		//double d = tanh_derivative(x[i], y[i]);
		double d = tanh_derivative(0.0, y[i]);
		double f = pre[i] * d;
		res[i] = f; // dy/dx
	}
	return res;
}

// ---- Pooling Node: 1D max ----

PoolMaxNode1D::PoolMaxNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), k(shape[0])
{
	nw = 0;
	assert(shape.size() == 1);
}

std::vector<int> PoolMaxNode1D::outShape(const std::vector<int>& inShape) const
{
	int n = (inShape[0] + k - 1) / k;
	return { n };
}

std::vector<double> PoolMaxNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = (x.size() + k - 1) / k;
	vector<double> res(n);
	for (size_t i = 0; i < n; ++i) {
		double v = x[i*k];
		size_t limit = min((i + 1)*k, x.size());
		for (size_t j = i * k + 1; j < limit; ++j)
			v = max(v, x[j]);
		res[i] = v;
	}
	return res;
}

std::vector<double> PoolMaxNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// no weight -> no change on <grad>
	// if argmax(x[1],...,x[n]) = i , then dy/dx = 1.0 and 0 for others
	const size_t ny = y.size();
	vector<double> res(x.size(), 0.0);
	for (size_t i = 0; i < ny; ++i) {
		size_t limit = min((i + 1)*k, x.size());
		for (size_t j = i * k; j < limit; ++j) {
			if (x[j] == y[i])
				res[j] = pre[i];
		}
	}
	return res;
}

// ---- Pooling Node: 2D max ----

PoolMaxNode2D::PoolMaxNode2D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), m(shape[1]), k1(shape[2]), k2(shape[3]),
	on((n + k1 - 1) / k1), om((m + k2 - 1) / k2)
{
	assert(shape.size() == 4);
	nw = 0;
}

std::vector<int> PoolMaxNode2D::outShape(const std::vector<int>& inShape) const
{
	return { static_cast<int>(on), static_cast<int>(om) };
}

std::vector<double> PoolMaxNode2D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	vector<double> res(on * om, numeric_limits<double>::lowest());
	int p = 0;
	for(int i = 0; i < n; ++i){
		int p1 = i / k1; // in range [0, on)
		int off = p1 * om;
		for(int j = 0; j < m; ++j){
			int p2 = j / k2; // in range [0, om)
			res[off + p2] = max(res[off + p2], x[p]);
			++p;
		}
	}
	return res;
}

std::vector<double> PoolMaxNode2D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// no weight -> no change on <grad>
	// if argmax(x[1],...,x[n]) = i , then dy/dx = 1.0 and 0 for others
	assert(y.size() == on * om);
	vector<double> res(x.size(), 0.0);
	int p = 0;
	for(int i = 0; i < n; ++i){
		int p1 = i / k1; // in range [0, on)
		int off = p1 * om;
		for(int j = 0; j < m; ++j){
			int p2 = j / k2; // in range [0, om)
			if(y[off + p2] == x[p]){
				res[p] = pre[p];
			}
			++p;
		}
	}
	return res;
}

// ---- Pooling Node: 1D min ----

PoolMinNode1D::PoolMinNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), k(shape[0])
{
	nw = 0;
	assert(shape.size() == 1);
}

std::vector<int> PoolMinNode1D::outShape(const std::vector<int>& inShape) const
{
	return std::vector<int>();
}

std::vector<double> PoolMinNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t n = (x.size() + k - 1) / k;
	vector<double> res(n);
	for(size_t i = 0; i < n; ++i) {
		double v = x[i*k];
		size_t limit = min((i + 1)*k, x.size());
		for(size_t j = i * k + 1; j < limit; ++j)
			v = min(v, x[j]);
		res[i] = v;
	}
	return res;
}

std::vector<double> PoolMinNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// no weight -> no change on <grad>
	// if argmin(x[1],...,x[n]) = i , then dy/dx = 1.0 and 0 for others
	const size_t ny = y.size();
	vector<double> res(x.size(), 0.0);
	for(size_t i = 0; i < ny; ++i) {
		size_t limit = min((i + 1)*k, x.size());
		for(size_t j = i * k; j < limit; ++j) {
			if(x[j] == y[i])
				res[j] = pre[i];
		}
	}
	return res;
}

// ---- Pooling Node: 2D min ----

PoolMinNode2D::PoolMinNode2D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), n(shape[0]), m(shape[1]), k1(shape[2]), k2(shape[3]),
	on((n + k1 - 1) / k1), om((m + k2 - 1) / k2)
{
	assert(shape.size() == 4);
	nw = 0;
}

std::vector<int> PoolMinNode2D::outShape(const std::vector<int>& inShape) const
{
	return { static_cast<int>(on), static_cast<int>(om) };
}

std::vector<double> PoolMinNode2D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	vector<double> res(on * om, numeric_limits<double>::max());
	int p = 0;
	for(int i = 0; i < n; ++i){
		int p1 = i / k1; // in range [0, on)
		int off = p1 * om;
		for(int j = 0; j < m; ++j){
			int p2 = j / k2; // in range [0, om)
			res[off + p2] = min(res[off + p2], x[p]);
			++p;
		}
	}
	return res;
}

std::vector<double> PoolMinNode2D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	// no weight -> no change on <grad>
	// if argmin(x[1],...,x[n]) = i , then dy/dx = 1.0 and 0 for others
	assert(y.size() == on * om);
	vector<double> res(x.size(), 0.0);
	int p = 0;
	for(int i = 0; i < n; ++i){
		int p1 = i / k1; // in range [0, on)
		int off = p1 * om;
		for(int j = 0; j < m; ++j){
			int p2 = j / k2; // in range [0, om)
			if(y[off + p2] == x[p]){
				res[p] = pre[p];
			}
			++p;
		}
	}
	return res;
}
