#pragma once
#include <string>
#include <vector>

enum struct NodeTypeGeneral {
	None, // for error
	Input, Trans,
	Sum,
	Conv, Recr,
	Pool, Act,
	FC
};

enum struct NodeType {
	None, // for error
	Input, Trans, 
	WeightedSum,
	Conv1D, Conv2D, Conv3D,
	RecrSig, RecrTanh, //RecrLSTM, RecrGate
	PoolMax1D, PoolMax2D, PoolMax3D,
	PoolMin1D, PoolMin2D, PoolMin3D,
	ActRelu, ActSigmoid, ActTanh,
	FC
};

using feature_t = std::vector<double>;

struct NodeBase {
	const size_t off;
	const std::vector<int> param;
	size_t nw;
	// offset: the offset of weights in the flatten <w>.
	// param: the structure parameter of the node
	NodeBase(const size_t offset, const std::vector<int>& param);
	size_t nweight() const;

	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	virtual void reset(); // reset internal state

	virtual feature_t predict(const feature_t& x, const std::vector<double>& w) = 0;
	// input: x, w, y, product of previous partial gradients.
	// pre-condition: predict(x,w) == y && y.size() == pre.size()
	// action 1: update corresponding entries of global <grad> vector (add: pre * dy/dw)
	// action 2: output product of all partial gradient (set to: pre * dy/dx)
	//           o[i] = pre * dy/dx[i] = sum_j (pre[j] * dy[j]/dx[i])
	// post-condition: result.size() == x.size() && nw == # of entries touched in <grad>
	virtual feature_t gradient(std::vector<double>& grad, const feature_t& x,
		const std::vector<double>& w, const feature_t& y, const feature_t& pre) = 0;

	virtual std::vector<feature_t> predict(const std::vector<feature_t>& x, const std::vector<double>& w);
	virtual std::vector<feature_t> gradient(std::vector<feature_t>& grad, const std::vector<feature_t>& x,
		const std::vector<double>& w, const feature_t& y, const std::vector<feature_t>& pre);
};

/*
// merge all <k> features (n-dimension) into one value, activate with sigmoid
// not a typical node.
// k*n => 1
// vector: y_{1*1} = sum ( W_{k*n} * x_{k*n} )
// individual: y = max_{i:0~k,j:0~n} ( W[i,j]*x[i,j] )
struct FCNode
	: public NodeBase
{
	FCNode(const size_t offset, const std::vector<int>& shape); // shape = {k,n}
	virtual std::vector<int> outShape(const std::vector<int>& inShape) const;
	// dummy
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) {
		return {};
	}
	// dummmy
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre) {
		return {};
	}
	// intput are k 1D-vectors. output is a scalar.
	double predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w);
	std::vector<std::vector<double>> gradient(std::vector<double>& grad, const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const double& y, const double& pre);
};
*/

NodeBase* generateNode(NodeType type, const size_t offset, const std::vector<int>& shape);
