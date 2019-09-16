#pragma once
#include "NodeBase.h"
#include <string>
#include <vector>
#include <tuple>
#include <functional>

struct VectorNetwork {
	using feature_t = std::vector<double>;

	// structure info
	int nLayer;
	std::vector<int> nNodeLayer;
	std::vector<NodeType> typeLayer;

	// node info
	std::vector<std::vector<int>> shapeNode; // the shape parameter of the node in each layer

	// feature info
	// the output at layer i is a matrix with shape (nFeatureLayer[i]*lenFeatureLayer[i])
	std::vector<int> numFeatureLayer; // the number of feature of layer i
	std::vector<int> lenFeatureLayer; // the length of a feature at layer i (=shpFeatureLayer[i].size())
	std::vector<std::vector<int>> shpFeatureLayer; // the actual shape of each feature at layer i

	// weight info
	std::vector<int> nWeightNode; // # of weight for a node at layer i
	std::vector<int> weightOffsetLayer; // weight offset of the the first node at layer i

	std::vector<std::vector<NodeBase*>> nodes;

	std::vector<std::vector<std::vector<double>>> mid; // intermediate result for each layer-feature-dimension
private:
	// function of the gradient of loss function. calculate gradient for each p entry
	// First Arg: predicted value. Second Arg: expected value
	std::function<feature_t(const feature_t& p, const feature_t& y)> fgl;
public:
	// R"((\d+(?:[\*x]\d+)*))"
	std::string getRegShape() const;
	// "[,-]"
	std::string getRegSep() const;
	// ",-"
	std::string getRawSep() const;
public:
	void init(const std::string& param); // calls parse and build
	// parse the parameter string into structure info
	// require: layers separated with "," or "-". Detailed parameters separated with ":"
	std::vector<std::tuple<int, NodeTypeGeneral, std::string>> parse(const std::string& param);
	// use the input structure info to build up the network
	void build(const std::vector<std::tuple<int, NodeTypeGeneral, std::string>>& structure);

	void bindGradLossFunc(std::function<feature_t(const feature_t& p, const feature_t& y)> glFun);
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w);
	std::vector<double> forward(const std::vector<double>& x, const std::vector<double>& w);
	std::vector<double> backward(
		const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y);
	std::vector<double> gradient(
		const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y);

	~VectorNetwork();

private:
	static std::vector<int> getShape(const std::string& str);
	static int getSize(const std::vector<int>& shape);
	static int getSize(const std::string& str);

	void createLayerInput(const size_t i, const std::vector<int>& shape);
	void createLayerAct(const size_t i, const std::string& type);
	void createLayerSum(const size_t i, const int n);
	void createLayerConv(const size_t i, const int n, const std::vector<int>& shape);
	void createLayerPool(const size_t i, const std::string& type, const std::vector<int>& shape);
	void createLayerRecr(const size_t i, const int n, const std::string& actType, const std::vector<int>& oshape);
	void createLayerFC(const size_t i, const int n);

	// set all data members and generate all nodes for layer i
	// precondition: weightOffsetLayer[i]
	void coreCreateLayer(const size_t i, const NodeType type, const int n, const std::vector<int>& shape);
	// precondition: typeLayer[i], nNodeLayer[i], shapeNode[i], weightOffsetLayer[i]
	// postcondition: nWeightNode[i], nodes[i], weightOffsetLayer[i+1]
	void createNodesForLayer(const size_t i);
};
