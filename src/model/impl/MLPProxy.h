#pragma once
#include <vector>

struct MLPProxyLayer;
struct MLPProxyNode;

struct MLPProxy {
	void init(const std::vector<int>& nNodes);
	int position(const int layer, const int from, const int to) const;
	int lengthParameter() const;

	void bind(const std::vector<double>* w);
	double get(const int layer, const int from, const int to) const;

	MLPProxyLayer getLayerProxy(const int l) const;
	MLPProxyLayer operator[](const int l) const;

	int nLayer;
	std::vector<int> nNodeLayer; // (layers) entries, v[i] = # of real nodes (except the constant-offset node)
	int nNodeTotal;
	// a dummy node is add to each layer excpet the last one, to perform the offset.
	std::vector<int> nWeightLayer; // (layers-1) entries, v[i] = (nNodeLayer[i]+1)*nNodeLayer[i+1]
	std::vector<int> nWeightLayerOffset; // (layers) entries, v[i] = sum nWeightLayer[j] for j in [0,i)
	int nWeightTotal;
private:
	const std::vector<double>* w;
};

struct MLPProxyLayer{
	// given the number of all weight-related nodes
	MLPProxyLayer(const int offset, const int nPrev, const int nNext,
		const std::vector<double>* w = nullptr)
		: offset(offset), n(nPrev), m(nNext), w(w)
	{}
	int position(const int from, const int to) const;

	void bind(const std::vector<double>* w);
	double get(const int from, const int to) const;

	MLPProxyNode getNodeProxyForward(const int i) const;
	MLPProxyNode getNodeProxyBackward(const int j) const;
	MLPProxyNode operator[](const int l) const; // same as forward
private:
	int offset;
	int n, m;
	const std::vector<double>* w;
};

struct MLPProxyNode{
	MLPProxyNode(const int offset, const int step, const std::vector<double>* w = nullptr)
		: offset(offset), step(step), w(w)
	{}
	int position(const int to) const;

	void bind(const std::vector<double>* w);
	double get(const int p) const;
	double operator[](const int p) const;
private:
	int offset;
	int step;
	const std::vector<double>* w;
};
