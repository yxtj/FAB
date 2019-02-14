#pragma once
#include "NodeBase.h"

// exactly repeat the intput
// n => n
// shape: y_{n} = x_{n}
// individual: y[i] = x[i]
struct InputNode
	: public NodeBase
{
	InputNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// convolution only, no activation
// n => n-k+1
// shape: y_{n-k+1} = conv(x_{n} , W_{k})
// individual: y[i] = sum_{j:0~k} (x[i+j] * W[j]) + b[i]
struct ConvNode1D
	: public NodeBase
{
	// y_{n-k+1}
	const int k;
	ConvNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// recurrent only, no activation
// n => k
// shape: y_{k*1} = W_{k*n} * x_{n*1} + U_{k*k} * y_{k*1} + b_{k*1}
// individual: y[i] = sum_{j:0~n} (W[i,j]*x[j]) + sum_{j:0~k} (U[i,j]*y[j]) + b[i]
struct RecuNode
	: public NodeBase
{
	const int n, k;
	std::vector<double> last;
	RecuNode(const size_t offset, const std::vector<int>& shape); // shape = {n,k}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// n => n
// shape: y_{n} = relu(x_{n})
// individual: y[i] = relu(x[i])
struct ReluNode
	: public NodeBase
{
	ReluNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// n => n
// shape: y_{n} = sigmoid(x_{n})
// individual: y[i] = sigmoid(x[i])
struct SigmoidNode
	: public NodeBase
{
	SigmoidNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// n => n/k ; more precisely ceil(n/k)
// shape: y_{n/k} = max(x_{n})
// individual: y[i] = max_{j:0~k} ( x[i*k+j] )
struct PoolMaxNode1D
	: public NodeBase
{
	PoolMaxNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k}
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// merge all features into one value, activate with sigmoid
// k*n => 1
// shape: y_{1*1} = sum ( W_{k*n} * x_{k*n} )
// individual: y = max_{i:0~k,j:0~n} ( W[i,j]*x[i,j] )
struct FCNode1D
	: public NodeBase
{
	FCNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k,n}
	// dummy
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) {
		return {};
	}
	// dummmy
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre) {
		return {};
	}
	// intput are k 1D vectors. output is a scalar.
	double predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w);
	std::vector<std::vector<double>> gradient(std::vector<double>& grad, const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const double& y, const double& pre);
};
