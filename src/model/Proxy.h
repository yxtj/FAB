#pragma once
#include <string>
#include <vector>

enum struct LayerType { Input, FC, Conv, Pool, Relu, Sigmoid, Tanh };

struct Proxy {
    void init(const std::string& param);
    int nLayer;
    std::vector<int> nNodeLayer;
    std::vector<LayerType> typeLayer;
    std::vector<std::vector<int>> iShape, oShape; // shape of input and output of a single node at each layer

private:
    std::vector<int> getShape(const std::string& str);
    int getSize(const std::vector<int>& shape);
};

struct NodeProxy {
    virtual double predict(const std::vector<double>& w, const std::vector<double>& x) = 0;
    virtual void gradient(std::vector<double>& grad,
        const std::vector<double>& w, const std::vector<double>& error) = 0;
};

