#include "NodeBase.h"
#include "NodeImpl.h"
#include "math/activation_func.h"

using namespace std;

// ---- NodeBase ----

NodeBase::NodeBase(const size_t offset, const std::vector<int>& shape)
	: off(offset), shape(shape)
{
	if(shape.empty()){
		nw = 0;
	}else{
		nw = 1;
		for(auto& v : shape)
			nw *= v;
	}
}

size_t NodeBase::nweight() const
{
	return nw;
}

std::vector<int> NodeBase::outShape(const std::vector<int>& inShape) const
{
	return inShape;
}

// ---- Fully-Connected Node ----

FCNode::FCNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw += 1;
}

std::vector<int> FCNode::outShape(const std::vector<int>& inShape) const
{
	return { 1 };
}

double FCNode::predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	double res = 0.0;
	size_t p = off;
	for(size_t i = 0; i < n1; ++i) {
		for(size_t j = 0; j < n2; ++j)
			res += x[i][j] * w[p++];
	}
	return sigmoid(res + w[p]);
}

std::vector<std::vector<double>> FCNode::gradient(
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
	for(size_t i = 0; i < n1; ++i) {
		for(size_t j = 0; j < n2; ++j) {
			grad[p] = x[i][j] * f; // pre * dy/dw
			pg[i][j] = w[p] * f; // pre * dy/dx
			++p;
		}
	}
	grad[p] += f; // the constant offset
	return pg;
}

NodeBase* generateNode(NodeType type, const size_t offset, const std::vector<int>& shape){
	NodeBase* p = nullptr;
	switch(type)
	{
	case NodeType::Input:
		p = new InputNode(offset, shape);
		break;
	case NodeType::WeightedSum:
		p = new WeightedSumNode(offset, shape);
		break;
	case NodeType::ActRelu:
		p = new ReluNode(offset, shape);
		break;
	case NodeType::ActSigmoid:
		p = new SigmoidNode(offset, shape);
		break;
	case NodeType::ActTanh:
		p = new TanhNode(offset, shape);
		break;
	case NodeType::Conv1D:
		p = new ConvNode1D(offset, shape);
		break;
	case NodeType::Conv2D:
		p = new ConvNode2D(offset, shape);
		break;
	case NodeType::RecrSig:
		p = new RecurrentSigmoidNode(offset, shape);
		break;
	case NodeType::RecrTanh:
		p = new RecurrentTanhNode(offset, shape);
		break;
	case NodeType::PoolMax1D:
		p = new PoolMaxNode1D(offset, shape);
		break;
	case NodeType::PoolMax2D:
		p = new PoolMaxNode2D(offset, shape);
		break;
	case NodeType::PoolMin1D:
		p = new PoolMinNode1D(offset, shape);
		break;
	case NodeType::PoolMin2D:
		p = new PoolMinNode2D(offset, shape);
		break;
	case NodeType::FC:
		p = new FCNode(offset, shape);
		break;
	}
	return p;
}
