#include "Proxy.h"
#include "util/Util.h"
#include "mathfunc.h"
#include <algorithm>
#include <regex>

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

ConvNode1D::ConvNode1D(const size_t offset)
    : off(offset)
{}

std::vector<double> ConvNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
    const size_t n = x.size() - w.size();
    const size_t k = w.size();
    std::vector<double> res(n + 1);
    double t = 0.0;
    for(size_t i=0; i<=n; ++i){
        double t = w[off+k];
        for(size_t j=0; j<k; ++j)
            t += x[j] * w[off+j];
        res[i] = t;
    }
    return res;
}

void ConvNode1D::gradient(std::vector<double>& grad, const std::vector<double>& y,
    const std::vector<double>& w, const std::vector<double>& error)
{
    
}

ReluNode::ReluNode(const size_t offset)
    : off(offset)
{}

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

void ReluNode::gradient(std::vector<double>& grad, const std::vector<double>& y,
    const std::vector<double>& w, const std::vector<double>& error)
{
    const size_t n = error.size();
    double s = 0.0;
    
}

// Sigmoid node

SigmoidNode::SigmoidNode(const size_t offset)
    : off(offset)
{}

std::vector<double> SigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
    const size_t n = x.size();
    std::vector<double> res(n);
    double t = 0.0;
    for(size_t i=0; i<n; ++i){
        res[i] = sigmoid(x[i] + w[off]);
    }
    return res;
}

void SigmoidNode::gradient(std::vector<double>& grad, const std::vector<double>& y,
    const std::vector<double>& w, const std::vector<double>& error)
{
    const size_t n = error.size();
    double s = 0.0;
    for(size_t i=0; i<n; ++i){
        s += sigmoid_derivative(y[i]);
    }
    grad[off] = s / n;
}

