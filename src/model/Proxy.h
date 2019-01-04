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

struct NodeProxy {
    virtual double predict(const std::vector<double>& w, const std::vector<double>& x) = 0;
    virtual void gradient(std::vector<double>& grad, const std::vector<double>& y,
        const std::vector<double>& w, const std::vector<double>& error) = 0;
};

struct ConvNode1D {
    const size_t off;
    explicit ConvNode1D(const size_t offset);
    std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
    void gradient(std::vector<double>& grad, const std::vector<double>& y,
        const std::vector<double>& w, const std::vector<double>& error);
};

struct ReluNode {
    const size_t off;
    explicit ReluNode(const size_t offset);
    std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
    void gradient(std::vector<double>& grad, const std::vector<double>& y,
        const std::vector<double>& w, const std::vector<double>& error);
};

struct SigmoidNode {
    const size_t off;
    explicit SigmoidNode(const size_t offset);
    std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
    void gradient(std::vector<double>& grad, const std::vector<double>& y,
        const std::vector<double>& w, const std::vector<double>& error);
};
