#include "Proxy.h"
#include "util/Util.h"
#include "mathfunc.h"
#include <algorithm>
#include <numeric>
#include <regex>
#include <cassert>

using namespace std;

void Proxy::init(const std::string& param){
    vector<string> strLayer = getStringList(param, "-");
    nLayer = strLayer.size();
    // raw string: R"(...)"
    string srShapeNode = R"((\d+(?:\*\d+)*))";
    //regex ri(srShapeNode); // input layer
    regex ra(R"((\d+),a,(sigmoid|relu|tanh))"); // activation layer
    regex rc(R"((\d+),c,)"+srShapeNode); // convolutional layer
    regex rp(R"((\d+),p,(max|mean|min),)"+srShapeNode); // pooling layer

    typeLayer.resize(nLayer);
    iShapeNode.resize(nLayer);
    oShapeNode.resize(nLayer);
    shapeLayer.resize(nLayer);
    dimLayer.resize(nLayer);

    typeLayer[0] = LayerType::Input;
    iShapeNode[0] = oShapeNode[0] = {1};
    shapeLayer[0] = getShape(strLayer[0]);
    dimLayer[0] = shapeLayer[0].size();
    for(size_t i = 1; i<strLayer.size(); ++i){
        smatch m;
        if(regex_match(strLayer[i], m, ra)){ // activation
            nNodeLayer[i] = stoi(m[1]);
            if(m[2] == "sigmoid")
                typeLayer[i] = LayerType::Sigmoid;
            else if(m[2] == "relu")
                typeLayer[i] = LayerType::Relu;
            else if(m[2] == "tanh")
                typeLayer[i] = LayerType::Tanh;
            iShapeNode[i] = {1};
            oShapeNode[i] = {1};
            shapeLayer[i] = shapeLayer[i-1];
            shapeLayer[i].insert(shapeLayer[i].begin(), nNodeLayer[i]);
            dimLayer[i] = shapeLayer[i].size();
        }else if(regex_match(strLayer[i], m, rc)){ // convolutional
            nNodeLayer[i] = stoi(m[1]);
            typeLayer[i] = LayerType::Conv;
            iShapeNode[i] = getShape(m[2]);
            oShapeNode[i] = {1};
            shapeLayer[i].push_back(nNodeLayer[i]);
            dimLayer[i] = 1 + dimLayer[i-1];
            int p = dimLayer[i] - iShapeNode[i].size();
            for(int j=0; j<dimLayer[i-1]; ++j){
                if(j<p)
                    shapeLayer[i].push_back(shapeLayer[i-1][j]);
                else
                    shapeLayer[i].push_back(shapeLayer[i-1][j] - iShapeNode[i][j] + 1);
            }
        }else if(regex_match(strLayer[i], m, rp)){ // pool
            nNodeLayer[i] = stoi(m[1]);
            typeLayer[i] = LayerType::Pool;
            string type = m[2];
            iShapeNode[i] = getShape(m[3]);
            oShapeNode[i] = {1};
        }
    }
}
    // std::vector<LayerType> ltype;
    // std::vector<vector<int>> iShapeNode, oShapeNode; // ShapeNode of input and output of each layer
    
std::vector<int> Proxy::getShape(const string& str){
    return getIntList(str, "*");
}

int Proxy::getSize(const std::vector<int>& ShapeNode){
    if(ShapeNode.empty())
        return 0;
    int r = 1;
    for(auto& v : ShapeNode)
        r*=v;
    return r;
}

// nodes

NodeBase::NodeBase(const size_t offset, const std::vector<int>& shape)
	: off(offset), shape(shape)
{
	nw = accumulate(shape.begin(), shape.end(), 1,
		[](double a, double b){return a * b; }
	);
}

size_t NodeBase::nweight() const
{
	return nw;
}

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
    for(size_t i=0; i<ny; ++i){
        double t = 0.0;
        for(size_t j=0; j<k; ++j)
            t += x[i+j] * w[off+j];
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
	for(size_t i = 0; i < nw; ++i){
		double t = 0.0;
		for(size_t j = 0; j < ny; ++j){
			t += pre[j] * x[j + i];
		}
		grad[off + i] = t;
	}
	// dy/dx
	std::vector<double> res(nx);
	for(size_t i = 0; i < nx; ++i){
		double t = 0.0;
		// cut the first and the last
		for(size_t j = (i < ny ? 0 : i - ny + 1); j < nw && i >= j; ++j){
			t += pre[i - j] * w[off + j];
		}
		res[i] = t;
	}
	return res;
}

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
    for(size_t i=0; i<n; ++i){
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
	for(size_t i = 0; i < n; ++i){
		double d = relu_derivative(x[i] + w[off]);
		s += pre[i] * d * x[i]; // dy/dw
		res[i] = pre[i] * d * w[off]; // dy/dx
	}
	grad[off] = s / n;
	return res;
}

// Sigmoid node

SigmoidNode::SigmoidNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{}

std::vector<double> SigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
    const size_t n = x.size();
    std::vector<double> res(n);
    for(size_t i=0; i<n; ++i){
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
    for(size_t i=0; i<n; ++i){
		//double d = sigmoid_derivative(x[i] + w[off], y[i]);
		double d = sigmoid_derivative(0.0, y[i]);
		s += pre[i] * d * x[i]; // dy/dw
		res[i] = pre[i] * d * w[off]; // dy/dx
    }
    grad[off] = s / n;
	return res;
}

// FC 1D node

FCNode1D::FCNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{}

double FCNode1D::predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	double res = 0.0;
	size_t p = off;
	for(size_t i = 0; i < n1; ++i){
		for(size_t j = 0; j < n2; ++j)
			res += x[i][j] * w[p++];
	}
	return sigmoid(res+w[p]);
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
	for(size_t i = 0; i < n1; ++i){
		for(size_t j = 0; j < n2; ++j){
			grad[p] = x[i][j] * f; // pre * dy/dw
			pg[i][j] = w[p] * f; // pre * dy/dx
			++p;
		}
	}
	grad[p] = f; // the constant offset
	return pg;
}
