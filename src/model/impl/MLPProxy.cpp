#include "MLPProxy.h"
using namespace std;

// -------- MLPProxy --------

void MLPProxy::init(const std::vector<int>& nNodes)
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

int MLPProxy::position(const int layer, const int from, const int to) const {
	return nWeightLayerOffset[layer] + (nNodeLayer[layer] + 1)*from + to;
}

int MLPProxy::lengthParameter() const
{
	return nWeightTotal;
}

void MLPProxy::bind(const std::vector<double>* w) {
	this->w = w;
}

double MLPProxy::get(const int layer, const int from, const int to) const
{
	int offset = position(layer, from, to);
	return (*w)[offset];
}

MLPProxyLayer MLPProxy::getLayerProxy(const int l) const
{
	return MLPProxyLayer(
		nWeightLayerOffset[l], nNodeLayer[l] + 1, nNodeLayer[l + 1], w);
}

MLPProxyLayer MLPProxy::operator[](const int l) const
{
	return getLayerProxy(l);
}

// -------- MLPProxyLayer --------

int MLPProxyLayer::position(const int from, const int to) const
{
	return offset + from * m + to;
}

void MLPProxyLayer::bind(const std::vector<double>* w)
{
	this->w = w;
}

double MLPProxyLayer::get(const int from, const int to) const
{
	return (*w)[position(from, to)];
}

MLPProxyNode MLPProxyLayer::getNodeProxyForward(const int i) const
{
	return MLPProxyNode(offset + i * m, 1, w);
}

MLPProxyNode MLPProxyLayer::getNodeProxyBackward(const int j) const
{
	return MLPProxyNode(offset + j, n, w);
}

MLPProxyNode MLPProxyLayer::operator[](const int i) const
{
	return getNodeProxyForward(i);
}

// -------- MLPProxyNode --------

int MLPProxyNode::position(const int to) const
{
	return offset + to * step;
}

void MLPProxyNode::bind(const std::vector<double>* w)
{
	this->w = w;
}

double MLPProxyNode::get(const int p) const
{
	return (*w)[position(p)];
}

double MLPProxyNode::operator[](const int p) const
{
	return get(p);
}
