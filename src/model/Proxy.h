#pragma once
#include <string>
#include <vector>

enum struct LayerType { Input, FC, Conv, Pool, Relu, Sigmoid, Tanh };

struct Proxy {
    void init(const std::string& param);
    int nLayer;
    std::vector<int> nNodeLayer;
    std::vector<LayerType> typeLayer;
    std::vector<std::vector<int>> iShapeNode, oShapeNode; // shape of input and output of a single node at each layer
    std::vector<std::vector<int>> shapeLayer;
    std::vector<int> dimLayer;

private:
    std::vector<int> getShape(const std::string& str);
    int getSize(const std::vector<int>& shape);
};

struct NodeBase{
	const size_t off;
	const std::vector<int> shape;
	size_t nw;
	NodeBase(const size_t offset, const std::vector<int>& shape);
	size_t nweight() const;
    virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) = 0;
	// input: x, w, y, product of previous partial gradients. 
	// pre-condition: predict(x,w) ==y && y.size() == pre.size()
	// action 1: update corresponding entries of global <grad> vector (pre[i] * dy/dw)
	// action 2: output product of all partial gradient (pre[i] * dy/dx)
	// post-condition: result.size() == x.size() && w.size() == # of entries touched in <grad>
    virtual std::vector<double> gradient(std::vector<double>& grad, const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre) = 0;
};

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
