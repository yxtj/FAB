#pragma once
#include "NodeBase.h"
#include <string>
#include <vector>

struct CNNProxy {
    void init(const std::string& param);
    int nLayer;
    std::vector<int> nNodeLayer;
    std::vector<NodeType> typeLayer;
	std::vector<std::vector<int>> unitNode; // the shape of input for 1 output entry of each node
	std::vector<std::vector<int>> shapeNode; // output shape of a single node at layer i
    std::vector<std::vector<int>> shapeLayer; // output shape of the whole layer i (nNodeLayer[i]*nFeatureLayer[i-1], shapeNode[i])

	std::vector<int> nFeatureLayer; // # of feature of layer i, = shapeLayer[i][0]
    std::vector<int> dimFeatureLayer; // = shapeLayer[i].size() -> shapeFeatureLayer[i].size()
	std::vector<std::vector<int>> shapeFeatureLayer; // the shape of each feature

	std::vector<int> nWeightNode; // # of weight for a node at layer i
	std::vector<int> weightOffsetLayer; // weight offset of the the first node at layer i

	std::vector<std::vector<NodeBase*>> nodes;
public:
	int lengthParameter() const;
private:
	void createLayerAct(const size_t i, const int n, const std::string& type);
	void createLayerConv(const size_t i, const int n, const std::vector<int>& shape);
	void createLayerPool(const size_t i, const int n, const std::string& type, const std::vector<int>& shape);
	void createLayerFC(const size_t i, const int n);

    std::vector<int> getShape(const std::string& str);
    int getSize(const std::vector<int>& shape);

	void setLayerParameter(const size_t i); // called after set shapeNode[i] && i>=1
	void generateNode(const size_t i); // called after all properties of i are set
};
