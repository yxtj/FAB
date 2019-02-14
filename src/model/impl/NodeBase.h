#pragma once
#include <string>
#include <vector>

enum struct NodeType {
	Input, FC, Conv, Recu,
	PoolMax, PoolMean, PoolMin,
	ActRelu, ActSigmoid, ActTanh
};

struct NodeBase {
	const size_t off;
	const std::vector<int> shape;
	size_t nw;
	// offset: the offset of weights in the flatten <w>.
	// shape: the structure parameter of the node (sometimes: shape of input for 1 output entry)
	NodeBase(const size_t offset, const std::vector<int>& shape);
	size_t nweight() const;

	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) = 0;
	// input: x, w, y, product of previous partial gradients.
	// pre-condition: predict(x,w) == y && y.size() == pre.size()
	// action 1: update corresponding entries of global <grad> vector (add: pre * dy/dw)
	// action 2: output product of all partial gradient (set to: pre * dy/dx)
	//           o[i] = pre * dy/dx[i] = sum_j (pre[j] * dy[j]/dx[i])
	// post-condition: result.size() == x.size() && nw == # of entries touched in <grad>
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre) = 0;
};
