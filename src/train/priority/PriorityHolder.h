#pragma once
#include <vector>
#include <utility>
#include <tuple>

class PriorityHolder{
public:
	virtual void init(const size_t size) = 0;
	virtual float get(const size_t id, const unsigned ver) = 0;
	virtual void set(const size_t id, const unsigned ver, const float prio) = 0;
	virtual void update(const size_t id, const unsigned ver, const float prio) = 0;
};

class PriorityHolderKeep : public PriorityHolder{
	std::vector<std::pair<float, unsigned>> priority;
public:
	virtual void init(const size_t size);
	virtual float get(const size_t id, const unsigned ver);
	virtual void set(const size_t id, const unsigned ver, const float prio);
	virtual void update(const size_t id, const unsigned ver, const float prio);
};

// p_n = p_o * exp(a * n)
class PriorityHolderExpLinear: public PriorityHolder {
	struct Item{
		float p; // 
		float a; // alpha
		unsigned n; // n-iteration
	};
	std::vector<Item> priority;
	float theta; // exponential decay
public:
	virtual void init(const size_t size);
	virtual float get(const size_t id, const unsigned ver);
	virtual void set(const size_t id, const unsigned ver, const float prio);
	virtual void update(const size_t id, const unsigned ver, const float prio);
};

// p_n = p_o * exp( (a*n + b) * n)
class PriorityHolderExpQuadratic : public PriorityHolder {
	std::vector<std::pair<float, unsigned>> priority;
	std::vector<std::tuple<float, float, unsigned>> old; // log(pn/po), tn^2-to^2, tn-to
	std::vector<std::pair<float, float>> factor;
	float alpha; // exponential decay
public:
	virtual void init(const size_t size);
	virtual float get(const size_t id, const unsigned ver);
	virtual void set(const size_t id, const unsigned ver, const float prio);
	virtual void update(const size_t id, const unsigned ver, const float prio);
};

