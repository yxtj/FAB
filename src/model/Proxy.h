#pragma once
#include <string>
#include <vector>

enum struct LayerType { Input, FC, Conv, PoolMax, PoolMean, PoolMin, ActRelu, ActSigmoid, ActTanh };

struct Proxy {
    void init(const std::string& param);
    int nLayer;
    std::vector<int> nNodeLayer;
    std::vector<LayerType> typeLayer;
	std::vector<std::vector<int>> unitNode; // the shape of input for 1 output entry of each node
	std::vector<std::vector<int>> shapeNode; // output shape of a single node at layer i
    std::vector<std::vector<int>> shapeLayer; // output shape of layer i
    std::vector<int> ndimLayer; // shapeLayer[i].size()
	std::vector<int> nWeightNode; // # of weight for a node at layer i
	std::vector<int> weightOffsetLayer; // weight offset of the the first node at layer i

	std::vector<std::vector<NodeBase*>> nodes;
private:
    std::vector<int> getShape(const std::string& str);
    int getSize(const std::vector<int>& shape);

	void setShapeLayer(int i); // called after set shapeNode[i]
	void generateNode(const int i); // called after all properties of i are set
};

struct NodeBase{
	const size_t off;
	const std::vector<int> shape;
	size_t nw;
	NodeBase(const size_t offset, const std::vector<int>& shape);
	size_t nweight() const;
    virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) = 0;
	// input: x, w, y, product of previous partial gradients. 
	// pre-condition: predict(x,w) == y && y.size() == pre.size()
	// action 1: update corresponding entries of global <grad> vector (pre[i] * dy/dw)
	// action 2: output product of all partial gradient (pre[i] * dy/dx)
	// post-condition: result.size() == x.size() && w.size() == # of entries touched in <grad>
    virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre) = 0;
};

struct InputNode
	: public NodeBase
{
	InputNode(const size_t offset, const std::vector<int>& shape);
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

// convolution only, no activation
struct ConvNode1D
	: public NodeBase
{
	const int k;
    ConvNode1D(const size_t offset, const std::vector<int>& shape);
    virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
    virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

struct ReluNode
	: public NodeBase
{
    ReluNode(const size_t offset, const std::vector<int>& shape);
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

struct SigmoidNode
	: public NodeBase
{
    SigmoidNode(const size_t offset, const std::vector<int>& shape);
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
        const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

struct PoolMaxNode1D
	: public NodeBase
{
	PoolMaxNode1D(const size_t offset, const std::vector<int>& shape);
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre);
};

struct FCNode1D
	: public NodeBase
{
	FCNode1D(const size_t offset, const std::vector<int>& shape);
	// dummy
	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w){
		return {};
	}
	// dummmy
	virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre){
		return {};
	}
	// intput are k 1D vectors. output is a scalar.
	double predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w);
	std::vector<std::vector<double>> gradient(std::vector<double>& grad, const std::vector<std::vector<double>>& x,
		const std::vector<double>& w, const double& y, const double& pre);
};
