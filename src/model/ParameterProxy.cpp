#include "ParameterProxy.h"
using namespace std;

// -------- ParameterProxy --------

void ParameterProxy::init(const std::vector<int>& nNodes)
{
	nLayer = static_cast<int>(nNodes.size());
	nNodeLayer.clear();
	nNodeLayer.reserve(nLayer);
	nNodeTotal = 0;
	for(auto v : nNodes){
		nNodeTotal += v;
		nNodeLayer.push_back(static_cast<size_t>(v));
	}
	// set nWeight
	nWeightTotal = 0;
	nWeightLayer.clear();
	nWeightLayer.reserve(nLayer - 1);
	nWeightLayerOffset.clear();
	nWeightLayerOffset.reserve(nLayer);
	for(size_t i = 0; i < nLayer - 1; ++i){
		auto v = (nNodeLayer[i] + 1)*nNodeLayer[i + 1];
		nWeightLayer.push_back(v);
		nWeightLayerOffset.push_back(nWeightTotal);
		nWeightTotal += v;
	}
	nWeightLayerOffset.push_back(nWeightTotal);
}

int ParameterProxy::position(const int layer, const int from, const int to) const {
	return nWeightLayerOffset[layer] + (nNodeLayer[layer] + 1)*from + to;
}

int ParameterProxy::lengthParameter() const
{
	return nWeightTotal;
}

void ParameterProxy::bind(const std::vector<double>* w) {
	this->w = w;
}

double ParameterProxy::get(const int layer, const int from, const int to) const
{
	int offset = position(layer, from, to);
	return (*w)[offset];
}

ParameterProxyLayer ParameterProxy::getLayerProxy(const int l) const
{
	return ParameterProxyLayer(
		nWeightLayerOffset[l], nNodeLayer[l] + 1, nNodeLayer[l + 1], w);
}

ParameterProxyLayer ParameterProxy::operator[](const int l) const
{
	return getLayerProxy(l);
}

// -------- ParameterProxyLayer --------

int ParameterProxyLayer::position(const int from, const int to) const
{
	return offset + from * m + to;
}

void ParameterProxyLayer::bind(const std::vector<double>* w)
{
	this->w = w;
}

double ParameterProxyLayer::get(const int from, const int to) const
{
	return (*w)[position(from, to)];
}

ParameterProxyNode ParameterProxyLayer::getNodeProxyForward(const int i) const
{
	return ParameterProxyNode(offset + i * m, 1, w);
}

ParameterProxyNode ParameterProxyLayer::getNodeProxyBackward(const int j) const
{
	return ParameterProxyNode(offset + j, n, w);
}

ParameterProxyNode ParameterProxyLayer::operator[](const int i) const
{
	return getNodeProxyForward(i);
}

// -------- ParameterProxyNode --------

int ParameterProxyNode::position(const int to) const
{
	return offset + to * step;
}

void ParameterProxyNode::bind(const std::vector<double>* w)
{
	this->w = w;
}

double ParameterProxyNode::get(const int p) const
{
	return (*w)[position(p)];
}

double ParameterProxyNode::operator[](const int p) const
{
	return get(p);
}
