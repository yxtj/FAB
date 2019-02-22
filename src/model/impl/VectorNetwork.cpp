#include "VectorNetwork.h"
#include "util/Util.h"
#include <regex>
#include <cassert>

using namespace std;

void VectorNetwork::init(const std::string& param){
	auto info = parse(param);
	build(info);
}

std::vector<std::tuple<int, NodeTypeGeneral, std::string>> VectorNetwork::parse(const std::string & param)
{
	// raw string format: R"(...)"
	string srShape = R"((\d+(?:[\*x]\d+)*))"; // v1[*v2[*v3[*v4]]], "*" can also be "x"
	regex ri(srShape); // input layer
	regex ra(R"((sig(?:moid)?|relu|tanh))"); // activation layer, i.e.: relu
	regex rc(R"((\d+):?c:?)" + srShape); // convolutional layer, i.e.: 3c3, 4:c4*5, 4:c:4*5x3
	regex rr(R"((\d+):?r:?)" + srShape); // recurrent layer, i.e.: 4r10, 6r:4*4
	regex rp(R"((max|min):?)" + srShape); // pooling layer, i.e.: max3*3, max:4
	regex rf(R"((\d+):?f?)"); // fully-connected layer, i.e.: 4f

	std::vector<std::tuple<int, NodeTypeGeneral, std::string>> res;
	vector<string> strParam = getStringList(param, ",-");
	for(string& str : strParam){
		smatch m;
		if(regex_match(str, m, ri)){
			res.emplace_back(1, NodeTypeGeneral::Input, m[1]);
		} else if(regex_match(str, m, ra)){
			res.emplace_back(1, NodeTypeGeneral::Act, m[1]);
		} else if(regex_match(str, m, rc)){
			res.emplace_back(stoi(m[1]), NodeTypeGeneral::Conv, m[2]);
		} else if(regex_match(str, m, rr)){
			res.emplace_back(stoi(m[1]), NodeTypeGeneral::Recr, m[2]);
		} else if(regex_match(str, m, rp)){
			res.emplace_back(1, NodeTypeGeneral::Pool, m[1]);
		} else if(regex_match(str, m, rf)){
			res.emplace_back(stoi(m[1]), NodeTypeGeneral::Act, "");
		} else{
			throw invalid_argument("unsupport network layer: " + str);
		}
	}

	return res;
}

void VectorNetwork::build(const std::vector<std::tuple<int, NodeTypeGeneral, std::string>>& structure)
{
	// data member initialize
	nLayer = static_cast<int>(structure.size());
	nNodeLayer.resize(nLayer);
	typeLayer.resize(nLayer);
	shapeNode.resize(nLayer);
	shapeNode.resize(nLayer);
	numFeatureLayer.resize(nLayer);
	nWeightNode.resize(nLayer);
	weightOffsetLayer.resize(nLayer + 1);
	nodes.resize(nLayer);

	// raw string format: R"(...)"
	string srShape = R"((\d+(?:[\*x]\d+)*))"; // v1[*v2[*v3[*v4]]]
	regex rShape(srShape);
	// create layers
	for(size_t i = 0; i < structure.size(); ++i){
		const auto& t = structure[i];
		const int n = get<0>(t);
		const string& parm = get<2>(t);
		smatch m;
		switch(get<1>(t))
		{
		case NodeTypeGeneral::Input:
			if(regex_match(parm, m, rShape))
				createLayerInput(i, getShape(m[1]));
			break;
		case NodeTypeGeneral::Act:
			createLayerAct(i, parm);
			break;
		case NodeTypeGeneral::Sum:
			if(regex_match(parm, m, rShape))
				createLayerSum(i, n);
		case NodeTypeGeneral::Conv:
			if(regex_match(parm, m, rShape))
				createLayerConv(i, n, getShape(m[1]));
			break;
		case NodeTypeGeneral::Recr:
			if(regex_match(parm, m, rShape))
				createLayerRecr(i, n, getShape(m[1]));
			break;
		case NodeTypeGeneral::Pool:
			if(regex_match(parm, m, regex("(max|min)[,:]?"+srShape)))
				createLayerPool(i, m[1], getIntList(m[1], "*x"));
			break;
		case NodeTypeGeneral::FC:
			createLayerFC(i, n);
			break;
		default:
			break;
		}
	}
}

void VectorNetwork::bindGradLossFunc(std::function<feature_t(const feature_t&p, const feature_t&y)> glFun)
{
	fgl = glFun;
}

int VectorNetwork::lengthParameter() const
{
	return weightOffsetLayer[nLayer];
}

VectorNetwork::~VectorNetwork()
{
	for(auto& layer : nodes){
		for(NodeBase* n : layer)
			delete n;
	}
}

std::vector<double> VectorNetwork::predict(
	const std::vector<double>& x, const std::vector<double>& w) const
{
	vector<vector<double>> input; // k features of n-dimension
	vector<vector<double>> output;
	input.push_back(x);
	// apart from the last FC layer, all nodes work on a single feature
	for(int i = 1; i < nLayer - 1; ++i){
		output.clear();
		// apply one node on each previous features repeatedly
		//assert(nFeatureLayer[i - 1] * nNodeLayer[i] == nFeatureLayer[i]);
		for(int j = 0; j < nNodeLayer[i]; ++j){
			for(int k = 0; k < numFeatureLayer[i - 1]; ++k){
				output.push_back(nodes[i][j]->predict(input[k], w));
			}
		}
		input = move(output);
	}
	// the last FC layer
	vector<double> res;
	int i = nLayer - 1;
	for(int j = 0; j < nNodeLayer[i]; ++j){
		FCNode* p = dynamic_cast<FCNode*>(nodes[i][j]);
		res.push_back(p->predict(input, w));
	}
	return res;
}

std::vector<double> VectorNetwork::gradient(
	const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y)
{
	// forward
	vector<vector<vector<double>>> mid; // layer -> feature -> value
	mid.reserve(nLayer); // intermediate result of all layers
	mid.push_back({ x });
	for(int i = 1; i < nLayer - 1; ++i){ // apart from the input and output (FC) layers
		const vector<vector<double>>& input = mid[i - 1];
		vector<vector<double>> output;
		// apply one node on each previous features repeatedly
		//assert(nFeatureLayer[i - 1] * nNodeLayer[i] == nFeatureLayer[i]);
		for(int j = 0; j < nNodeLayer[i]; ++j){
			for(int k = 0; k < numFeatureLayer[i - 1]; ++k){
				output.push_back(nodes[i][j]->predict(input[k], w));
			}
		}
		mid.push_back(move(output));
	}
	vector<FCNode*> finalNodes;
	{
		const vector<vector<double>>& input = mid[nLayer - 2];
		vector<double> output;
		for(int j = 0; j < nNodeLayer.back(); ++j){
			FCNode* p = dynamic_cast<FCNode*>(nodes.back()[j]);
			finalNodes.push_back(p);
			output.push_back(p->predict(input, w));
		}
		mid.push_back({ move(output) });
	}
	// Back Propagate
	vector<double> grad(w.size());
	vector<vector<double>> partial; // partial gradient
	size_t n = mid.back().size();
	//assert(mid.back().size() ==1 && y.size() == mid.back()[0].size());
	// BP (last layer)
	{
		const vector<double>& output = mid.back().front();
		vector<double> pg = fgl(output, y);
		for(size_t i = 0; i < y.size(); ++i){ // last FC layer
			FCNode* p = finalNodes[i];
			vector<vector<double>> temp = p->gradient(grad, mid[nLayer - 2], w, output[i], pg[i]);
			if(i == 0){
				partial = move(temp);
			} else{
				// partial += temp;
				for(size_t a = 0; a < temp.size(); ++a)
					for(size_t b = 0; b < temp[a].size(); ++b)
						partial[a][b] += temp[a][b];
			}
		}
	}
	// BP (0 to n-1 layer)
	for(int i = nLayer - 2; i > 0; --i){ // layer
		int oidx = 0;
		vector<vector<double>> newPartialGradient(numFeatureLayer[i]);
		for(int j = 0; j < nNodeLayer[i]; ++j){ // node
			for(int k = 0; k < numFeatureLayer[i - 1]; ++k){ // feature (input)
				const vector<double>& input = mid[i - 1][k];
				//oidx == j * numFeatureLayer[i - 1] + k;
				const vector<double>& output = mid[i][oidx];
				const vector<double>& pg = partial[oidx];
				++oidx;
				NodeBase* p = nodes[i][j];
				vector<double> npg = p->gradient(grad, input, w, output, pg);
				if(j == 0){
					newPartialGradient[k] = move(npg);
				} else{
					for(size_t a = 0; a < npg.size(); ++a)
						newPartialGradient[k][a] += npg[a];
				}
			}
		}
		partial = move(newPartialGradient);
	}
	return grad;
}

// ---- helper functions ----

std::vector<int> VectorNetwork::getShape(const std::string& str){
	return getIntList(str, "*x");
}

int VectorNetwork::getSize(const std::vector<int>& shape){
	if(shape.empty())
		return 0;
	int r = 1;
	for(auto& v : shape)
		r *= v;
	return r;
}

int VectorNetwork::getSize(const std::string & str)
{
	return getSize(getShape(str));
}

// ---- create functions ----

void VectorNetwork::createLayerInput(const size_t i, const std::vector<int>& shape){
	coreCreateLayer(i, NodeType::Input, 1, shape);
}

void VectorNetwork::createLayerAct(const size_t i,const std::string& type){
	NodeType ntp;
	if(type == "sigmoid")
		ntp = NodeType::ActSigmoid;
	else if(type == "relu")
		ntp = NodeType::ActRelu;
	else if(type == "tanh")
		ntp = NodeType::ActTanh;
	coreCreateLayer(i, ntp, 1, {});
}

void VectorNetwork::createLayerSum(const size_t i, const int n){
	coreCreateLayer(i, NodeType::WeightedSum, n, { lenFeatureLayer[i - 1] });
}

void VectorNetwork::createLayerConv(const size_t i, const int n, const std::vector<int>& shape){
	assert(shape.size() == shpFeatureLayer[i - 1].size());
	if(shape.size() == 1)
		coreCreateLayer(i, NodeType::Conv1D, n, shape);
}

void VectorNetwork::createLayerRecr(const size_t i, const int n, const std::vector<int>& oshape){
	coreCreateLayer(i, NodeType::RecrFully, n,
		{ lenFeatureLayer[i - 1], getSize(oshape) });
}

void VectorNetwork::createLayerPool(const size_t i, const std::string& type, const std::vector<int>& shape){
	NodeType ntp;
	if(type == "max"){
		if(shape.size() == 1)
			ntp = NodeType::PoolMax1D;
	} else if(type == "min"){
		if(shape.size() == 1)
			ntp = NodeType::PoolMin1D;
	}
	coreCreateLayer(i, ntp, 1, shape);
}

void VectorNetwork::createLayerFC(const size_t i, const int n){
	coreCreateLayer(i, NodeType::FC, n, { numFeatureLayer[i - 1], lenFeatureLayer[i - 1] });
}

void VectorNetwork::coreCreateLayer(const size_t i,
	const NodeType type, const int n, const std::vector<int>& shape)
{
	nNodeLayer[i] = n;
	typeLayer[i] = type;
	shapeNode[i] = shape;
	createNodesForLayer(i); // set nWeightNode, weightOffsetLayer and nodes
	numFeatureLayer[i] = numFeatureLayer[i - 1] * n;
	shpFeatureLayer[i] = nodes[i][0]->outShape(shpFeatureLayer[i - 1]);
	lenFeatureLayer[i] = getSize(shpFeatureLayer[i]);
}

void VectorNetwork::createNodesForLayer(const size_t i)
{
	int offset = weightOffsetLayer[i];
	vector<NodeBase*>& vec = nodes[i];
	for(int j = 0; j < nNodeLayer[i]; ++j){
		NodeBase* p = generateNode(typeLayer[i], offset, shapeNode[i]);
		int nw = static_cast<int>(p->nweight());
		offset += nw;
		nWeightNode[i] = nw;
		vec.push_back(p);
	}
	weightOffsetLayer[i + 1] = offset;
}