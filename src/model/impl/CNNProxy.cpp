#include "CNNProxy.h"
#include "NodeImpl.h"
#include "util/Util.h"
#include <regex>
#include <cassert>

using namespace std;

void CNNProxy::init(const std::string& param){
    vector<string> strLayer = getStringList(param, "-");
    nLayer = static_cast<int>(strLayer.size());
    // raw string format: R"(...)"
    string srShapeNode = R"((\d+(?:\*\d+)*))"; // v1[*v2[*v3[*v4]]]
    //regex ri(srShapeNode); // input layer
    regex ra(R"((\d+),a,(sigmoid|relu))"); // activation layer, i.e.: 1,a,relu
    regex rc(R"((\d+),c,)"+srShapeNode); // convolutional layer, i.e.: 4,c,4*4
    regex rp(R"((\d+),p,(max|mean|min),)"+srShapeNode); // pooling layer, i.e.: 1,p,max,3*3
	regex rf(R"((\d+),f)"); // fully-connected layer, i.e.: 4,f

	nNodeLayer.resize(nLayer);
    typeLayer.resize(nLayer);
    unitNode.resize(nLayer);
    shapeNode.resize(nLayer);
    shapeLayer.resize(nLayer);
    dimFeatureLayer.resize(nLayer);
	nFeatureLayer.resize(nLayer);
	nWeightNode.resize(nLayer);
	weightOffsetLayer.resize(nLayer + 1);
	nodes.resize(nLayer);

	// input (first layer)
	nNodeLayer[0] = 1;
    typeLayer[0] = NodeType::Input;
	unitNode[0] = { 1 };
	shapeNode[0] = shapeLayer[0] = getShape(strLayer[0]);
    dimFeatureLayer[0] = static_cast<int>(shapeLayer[0].size());
	nFeatureLayer[0] = 1;
	nWeightNode[0] = weightOffsetLayer[0] = 0;
    for(size_t i = 1; i<strLayer.size(); ++i){
        smatch m;
        if(regex_match(strLayer[i], m, ra)){ // activation
			createLayerAct(i, stoi(m[1]), m[2]);
        }else if(regex_match(strLayer[i], m, rc)){ // convolutional
			createLayerConv(i, stoi(m[1]), getShape(m[2]));
        }else if(regex_match(strLayer[i], m, rp)){ // pool
			createLayerPool(i, stoi(m[1]), m[2], getShape(m[3]));
		}else if(regex_match(strLayer[i], m, rf)){ // fully-connected
			createLayerFC(i, stoi(m[1]));
		} else if(i == strLayer.size() - 1 && regex_match(strLayer[i], m, regex(R"((\d+))"))){
			// fully-connected at the last layer
			createLayerFC(i, stoi(m[1]));
		} else{
			throw invalid_argument("Unsupported node parameter: " + strLayer[i]);
		}
    }
}
  
int CNNProxy::lengthParameter() const
{
	return weightOffsetLayer[nLayer];
}

void CNNProxy::createLayerAct(const size_t i, const int n, const std::string& type){
	nNodeLayer[i] = n;
	if(type == "sigmoid")
		typeLayer[i] = NodeType::ActSigmoid;
	else if(type == "relu")
		typeLayer[i] = NodeType::ActRelu;
	else if(type == "tanh")
		typeLayer[i] = NodeType::ActTanh;
	unitNode[i] = { 1 };
	shapeNode[i] = shapeNode[i - 1];
	setLayerParameter(i);
	generateNode(i);
}

void CNNProxy::createLayerConv(const size_t i, const int n, const std::vector<int>& shape){
	nNodeLayer[i] = n;
	typeLayer[i] = NodeType::Conv;
	unitNode[i] = shape;
	size_t p = shapeLayer[i - 1].size() - unitNode[i].size();
	assert(p == 0 || p == 1);
	for(size_t j = 0; j < unitNode[i].size(); ++j){
		shapeNode[i].push_back(shapeLayer[i - 1][p + j] - unitNode[i][j] + 1);
	}
	setLayerParameter(i);
	generateNode(i);
}

void CNNProxy::createLayerPool(const size_t i, const int n, const std::string& type, const std::vector<int>& shape){
	nNodeLayer[i] = n;
	if(type == "max")
		typeLayer[i] = NodeType::PoolMax;
	else if(type == "mean")
		typeLayer[i] = NodeType::PoolMean;
	else if(type == "min")
		typeLayer[i] = NodeType::PoolMin;
	unitNode[i] = shape;
	size_t p = shapeLayer[i - 1].size() - unitNode[i].size();
	assert(p == 0 || p == 1);
	for(size_t j = 0; j < unitNode[i].size(); ++j){
		int v = (shapeLayer[i - 1][p + j] + unitNode[i][j] - 1) / unitNode[i][j];
		shapeNode[i].push_back(v);
	}
	setLayerParameter(i);
	generateNode(i);
}

void CNNProxy::createLayerFC(const size_t i, const int n){
	nNodeLayer[i] = n;
	typeLayer[i] = NodeType::FC;
	unitNode[i] = shapeLayer[i - 1];
	shapeNode[i] = { 1 };
	shapeLayer[i] = { nNodeLayer[i], 1 };
	dimFeatureLayer[i] = 2;
	generateNode(i);
}

std::vector<int> CNNProxy::getShape(const string& str){
    return getIntList(str, "*");
}

int CNNProxy::getSize(const std::vector<int>& ShapeNode){
    if(ShapeNode.empty())
        return 0;
    int r = 1;
    for(auto& v : ShapeNode)
        r*=v;
    return r;
}

void CNNProxy::setLayerParameter(const size_t i){
	if(shapeLayer[i - 1].size() == shapeNode[i].size()){
		nFeatureLayer[i] = nNodeLayer[i];
	}else if(shapeLayer[i - 1].size() == shapeNode[i].size() + 1){
		nFeatureLayer[i] = shapeLayer[i - 1][0] * nNodeLayer[i];
	} else{
		throw invalid_argument("shape between layer "
			+ to_string(i - 1) + " and " + to_string(i) + " does not match.");
	}
	shapeLayer[i].push_back(nFeatureLayer[i]);
	for(size_t j = 0; j < shapeNode[i].size(); ++j)
		shapeLayer[i].push_back(shapeNode[i][j]);
	dimFeatureLayer[i] = static_cast<int>(shapeLayer[i].size());
}

void CNNProxy::generateNode(const size_t i)
{
	int offset = weightOffsetLayer[i];
	vector<NodeBase*>& vec = nodes[i];
	for(int j = 0; j < nNodeLayer[i]; ++j){
		NodeBase* p = nullptr;
		switch(typeLayer[i])
		{
		case NodeType::ActRelu:
			p = new ReluNode(offset, unitNode[i]);
			break;
		case NodeType::ActSigmoid:
			p = new SigmoidNode(offset, unitNode[i]);
			break;
		case NodeType::Conv:
			p = new ConvNode1D(offset, unitNode[i]);
			break;
		case NodeType::PoolMax:
			p = new PoolMaxNode1D(offset, unitNode[i]);
			break;
		case NodeType::FC:
			p = new FCNode1D(offset, unitNode[i]);
			break;
		default:
			throw invalid_argument("try to generate an unsupported node type.");
			break;
		}
		int nw = static_cast<int>(p->nweight());
		offset += nw;
		nWeightNode[i] = nw;
		vec.push_back(p);
	}
	weightOffsetLayer[i + 1] = offset;
}
