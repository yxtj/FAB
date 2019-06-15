#include "PriorityHolder.h"
using namespace std;

void PriorityHolder::init(const size_t size)
{
	priority.resize(size);
}

float PriorityHolder::get(const int id, const unsigned ver)
{
	return priority[id].first;
}

void PriorityHolder::set(const int id, const unsigned ver, const float prio)
{
	priority[id].first = prio;
}

void PriorityHolder::update(const int id, const unsigned ver, const float prio)
{
	priority[id].first = prio;
}

// exp-linear

void PriorityHolderExpLinear::init(const size_t size)
{
	priority.resize(size);
	factor.resize(size, 1.0f);
}

float PriorityHolderExpLinear::get(const int id, const unsigned ver)
{
	return priority[id].first * exp(factor[id] * (ver - priority[id].second));
}

void PriorityHolderExpLinear::set(const int id, const unsigned ver, const float prio)
{
	priority[id] = make_pair(prio, ver);
}

void PriorityHolderExpLinear::update(const int id, const unsigned ver, const float prio)
{
	// priority->p2, parameter->p1
	// log(p1/p2) = a(n1-n2)
	float dp = log(prio / priority[id].first);
	unsigned dn = ver - priority[id].second;
	factor[id] = dp / dn;
	priority[id] = make_pair(prio, ver);
}

// exp-twice

void PriorityHolderExpTwice::init(const size_t size)
{
	priority.resize(size);
	old.resize(size);
	factor.resize(size, make_pair(0.0f, 1.0f));
}

float PriorityHolderExpTwice::get(const int id, const unsigned ver)
{
	auto d1 = ver - priority[id].second;
	if(d1 == 0)
		return priority[id].first;
	auto d2 = ver*ver - priority[id].second*priority[id].second;
	return priority[id].first * exp(factor[id].first*d2 + factor[id].second*d1);
}

void PriorityHolderExpTwice::set(const int id, const unsigned ver, const float prio)
{
	old[id] = make_tuple(prio, 0.0f, 0);
	priority[id] = make_pair(ver, prio);
}

void PriorityHolderExpTwice::update(const int id, const unsigned ver, const float prio)
{
	// old->p3, priority->p2, parameter->p1
	// log(p1/p2) = a(n1^2-n2^2) + b(n1-n2)
	// log(p2/p3) = a(n2^2-n3^2) + b(n2-n3)
	// matrix: [pn,po]^T = [[dn2,dn1],[do2,do1]] * [a,b]^T
	unsigned dn1 = ver - priority[id].second;
	if(dn1 == 0){
		return;
	} 
	float dn2 = static_cast<float>(ver)*ver - static_cast<float>(priority[id].second)*priority[id].second;
	float pn = log(prio / priority[id].first);
	unsigned do1 = std::get<2>(old[id]);
	if(do1 == 0){
		// use exp-linear
		float b = pn / dn1;
		factor[id] = make_pair(0.0f, b);
	} else{
		float do2 = std::get<1>(old[id]);
		float po = std::get<0>(old[id]);
		// calculate parameter
		float a = (pn*do1 - po * dn1) / (dn2*do1 - dn1 * do2);
		float b = (pn*do2 - po * dn2) / (dn1*do2 - do1 * dn2);
		factor[id] = make_pair(a, b);
	}
	// set buffer
	old[id] = make_tuple(pn, dn2, dn1);
}
