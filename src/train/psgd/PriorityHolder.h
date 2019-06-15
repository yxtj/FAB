#pragma once
#include <vector>
#include <utility>
#include <tuple>

class PriorityHolder{
public:
	std::vector<std::pair<float, unsigned>> priority;
	virtual void init(const size_t size);
	virtual float get(const int id, const unsigned ver);
	virtual void set(const int id, const unsigned ver, const float prio);
	virtual void update(const int id, const unsigned ver, const float prio);
};

// p_n = p_o * exp(a * n)
class PriorityHolderExpLinear: public PriorityHolder {
	std::vector<float> factor;
public:
	virtual void init(const size_t size);
	virtual float get(const int id, const unsigned ver);
	virtual void set(const int id, const unsigned ver, const float prio);
	virtual void update(const int id, const unsigned ver, const float prio);
};

// p_n = p_o * exp( (a*n + b) * n)
class PriorityHolderExpTwice : public PriorityHolder {
	std::vector<std::tuple<float, float, unsigned>> old; // log(pn/po), tn^2-to^2, tn-to
	std::vector<std::pair<float, float>> factor;
public:
	virtual void init(const size_t size);
	virtual float get(const int id, const unsigned ver);
	virtual void set(const int id, const unsigned ver, const float prio);
	virtual void update(const int id, const unsigned ver, const float prio);
};

