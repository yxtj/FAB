#pragma once
#include "NodeBase.h"

// exactly repeat the intput
// n => n
// vector: y_{n} = x_{n}
// individual: y[i] = x[i]
struct InputNode
	: public NodeBase
{
	InputNode(const size_t offset, const std::vector<int>& shape); // shape = {n}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// transform each input feature to <on> output features, each output of shape <ofshape>
// n => <on> * <ofshape>
struct TransNode
	: public NodeBase
{
	const int on; // number of output features for each input feature
	const std::vector<int> ofshape; // shape of each output feature
	TransNode(const size_t offset, const std::vector<int>& shape); // shape = {on, ofshape}
	// dummy feature-wise function
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w){}
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre){}
		
	virtual std::vector<feature_t> predict(const std::vector<feature_t>& x, const std::vector<double>& w);
	virtual std::vector<feature_t> gradient(std::vector<feature_t>& grad, const std::vector<feature_t>& x,
		const std::vector<double>& w, const std::vector<feature_t>& y, const std::vector<feature_t>& pre);
};

// do weighted summation
// n => k
// vector: y_{k*1} = W_{k*n} * x_{n*1} + b_{k*1}
// individual: y[i] = sum_{j:0~n} ( W[i,j] * x[j] ) + b[i]
struct WeightedSumNode
	: public NodeBase
{
	const int n, k;
	WeightedSumNode(const size_t offset, const std::vector<int>& shape); // shape = {n,k}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// merge all <k> features (n-dimension) into one value, activate with sigmoid
// not a typical node.
// k*n => 1
// vector: y_{1*1} = sum ( W_{k*n} * x_{k*n} )
// individual: y = max_{i:0~k,j:0~n} ( W[i,j]*x[i,j] )
struct FCNode
	: public NodeBase
{
	const int n, k;
	FCNode(const size_t offset, const std::vector<int>& shape); // shape = {k,n}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w); // dummy
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre); // dummy
	virtual std::vector<feature_t> predict(const std::vector<feature_t>& x, const std::vector<double>& w);
	virtual std::vector<feature_t> gradient(std::vector<feature_t>& grad, const std::vector<feature_t>& x,
		const std::vector<double>& w, const feature_t& y, const std::vector<feature_t>& pre);
};

// convolution only, no activation
// n => n-k+1
// vector: y_{n-k+1} = conv(x_{n} , W_{k}) + b_{1}
// individual: y[i] = sum_{j:0~k} (x[i+j] * W[j]) + b
struct ConvNode1D
	: public NodeBase
{
	// y_{n-k+1}
	const int k;
	ConvNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// convolution only, no activation
// n*m => (n-k1+1)*(m-k2+1)
// vector: y_{n-k1+1,m-k2+1} = conv(x_{n,m} , W_{k1,k2}) + b_{1}
// individual: y[i][j] = sum_{p1:0~k1,p2:0~k2} (x[i+p1][j+p2] * W[p1][p2]) + b
struct ConvNode2D
	: public NodeBase
{
	const int n, m;
	const int k1, k2;
	const int on, om;
	ConvNode2D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, k1, k2}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// convolution only, no activation
// n*m*p => (n-k1+1)*(m-k2+1)*(p-k3+1)
// vector: y_{n-k1+1,m-k2+1,p-k3+1} = conv(x_{n,m,p} , W_{k1,k2,k3}) + b_{1}
// individual: y[i][j][k] = sum_{p1:0~k1,p2:0~k2,p3:0~k3} (x[i+p1][j+p2][k+p3] * W[p1][p2][p3]) + b
struct ConvNode3D
	: public NodeBase
{
	const int n, m, p;
	const int k1, k2, k3;
	const int on, om, op;
	ConvNode3D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, p, k1, k2, k3}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// recurrent node base (no activation). use history state remembering last output
// n => k
// vector: y_{k*1} = act( W_{k*n} * x_{n*1} + U_{k*k} * y_{k*1} + b_{k*1} )
// individual: y[i] = act( sum_{j:0~n} (W[i,j]*x[j]) + sum_{j:0~k} (U[i,j]*y[j]) + b[i] )
struct RecurrentNodeBase
	: public NodeBase
{
	const int n, k;
	std::vector<double> last_pred, last_grad; // store the last output (k-dim) for both predict and gradient
	RecurrentNodeBase(const size_t offset, const std::vector<int>& shape); // shape = {n,k}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual void reset();
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);

	std::vector<double> predictCalcOnly(const std::vector<double>& x, const std::vector<double>& w);
};

// recurrent node using sigmoid activation
// n => k
// y = sigmoid( W * x + U * y + b )
struct RecurrentSigmoidNode
	: public RecurrentNodeBase
{
	RecurrentSigmoidNode(const size_t offset, const std::vector<int>& shape); // shape = {n,k}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// recurrent node using tanh activation
// n => k
// y = tanh( W * x + U * y + b )
struct RecurrentTanhNode
	: public RecurrentNodeBase
{
	RecurrentTanhNode(const size_t offset, const std::vector<int>& shape); // shape = {n,k}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// n => n
// vector: y_{n} = relu(x_{n})
// individual: y[i] = relu(x[i])
struct ReluNode
	: public NodeBase
{
	ReluNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// n => n
// vector: y_{n} = sigmoid(x_{n})
// individual: y[i] = sigmoid(x[i])
struct SigmoidNode
	: public NodeBase
{
	SigmoidNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// n => n
// vector: y_{n} = tanh(x_{n})
// individual: y[i] = tanh(x[i])
struct TanhNode
	: public NodeBase
{
	TanhNode(const size_t offset, const std::vector<int>& shape); // shape = {1}
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the max value
// n => n/k ; more precisely ceil(n/k)
// vector: y_{n/k} = max(x_{n})
// individual: y[i] = max_{j:0~k} ( x[i*k+j] )
struct PoolMaxNode1D
	: public NodeBase
{
	const size_t k;
	PoolMaxNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the max value
// n,m => n/k1,m/k2 ; more precisely ceil(n/k1),ceil(m/k2)
// vector: y_{n/k1,m/k2} = max(x_{n,m})
// individual: y[i][j] = max_{p1:0~k1,p2:0~k2} ( x[i*k1+p1][j*k2+p2] )
struct PoolMaxNode2D
	: public NodeBase
{
	const int n, m;
	const int k1, k2;
	const int on, om;
	PoolMaxNode2D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, k1, k2}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the max value
// n,m,p => n/k1,m/k2,p/k3 ; more precisely ceil(n/k1),ceil(m/k2),ceil(p/k3)
// vector: y_{n/k1,m/k2,p/k3} = max(x_{n,m,p})
// individual: y[i][j][k] = max_{p1:0~k1,p2:0~k2,p3:0~k3} ( x[i*k1+p1][j*k2+p2][k*k3+p3] )
struct PoolMaxNode3D
	: public NodeBase
{
	const int n, m, p;
	const int k1, k2, k3;
	const int on, om, op;
	PoolMaxNode3D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, p, k1, k2, k3}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the min value
// n => n/k ; more precisely ceil(n/k)
// vector: y_{n/k} = min(x_{n})
// individual: y[i] = min_{j:0~k} ( x[i*k+j] )
struct PoolMinNode1D
	: public NodeBase
{
	const size_t k;
	PoolMinNode1D(const size_t offset, const std::vector<int>& shape); // shape = {k}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the min value
// n,m => n/k1,m/k2 ; more precisely ceil(n/k1),ceil(m/k2)
// vector: y_{n/k1,m/k2} = min(x_{n,m})
// individual: y[i][j] = min_{p1:0~k1,p2:0~k2} ( x[i*k1+p1][j*k2+p2] )
struct PoolMinNode2D
	: public NodeBase
{
	const int n, m;
	const int k1, k2;
	const int on, om;
	PoolMinNode2D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, k1, k2}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};

// get the min value
// n,m,p => n/k1,m/k2,p/k3 ; more precisely ceil(n/k1),ceil(m/k2),ceil(p/k3)
// vector: y_{n/k1,m/k2,p/k3} = min(x_{n,m,p})
// individual: y[i][j][k] = min_{p1:0~k1,p2:0~k2,p3:0~k3} ( x[i*k1+p1][j*k2+p2][k*k3+p3] )
struct PoolMinNode3D
	: public NodeBase
{
	const int n, m, p;
	const int k1, k2, k3;
	const int on, om, op;
	PoolMinNode3D(const size_t offset, const std::vector<int>& shape); // shape = {n, m, p, k1, k2, k3}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual feature_t predict(const feature_t& x, const std::vector<double>& w);
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre);
};
