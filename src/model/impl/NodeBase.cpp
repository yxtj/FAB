#include "NodeBase.h"
#include "NodeImpl.h"

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
	case NodeType::RecrFully:
		p = new RecurrentNode(offset, shape);
		break;
	case NodeType::PoolMax1D:
		p = new PoolMaxNode1D(offset, shape);
		break;
	case NodeType::PoolMin1D:
		p = new PoolMinNode1D(offset, shape);
		break;
	case NodeType::FC:
		p = new FCNode(offset, shape);
		break;
	}
	return p;
}
