#pragma once
#include "../Trainer.h"
#include "../PSGD.h"
#include "../priority/PriorityHolder.h"
#include "../priority/PriorityDumper.h"

class PSGD_log : public PSGD
{
	// parameters
	double rate = 1.0;
	double topRatio = 1.0;
	double renewRatio = 0.01;
	// priority
	enum struct PriorityType{
		Projection, // data-point i: pi=gi*avg(g)
		Length, // pi=gi*gi
	};
	PriorityType prioInitType = PriorityType::Length;
	PriorityType prioType = PriorityType::Projection;
	enum struct DecayType{
		Keep, // time i, all parameter a,b,c are dynamic: pi=a
		ExpLinear, // pi=exp(a+bi), pi=pj*exp(b(i-j))
		ExpQuadratic, // pi=exp(a+bi+ci^2), pi=pj*exp(b(i-j)+c(i^2-j^2))
	};
//	std::vector<float> priority;
	PriorityHolder* prhd = nullptr;
	PriorityDumper pdump;
	std::vector<int> priorityIdx; // for top-k
	unsigned wver; // parameter version

	// gradient
	std::vector<double> avgGrad;
	// variations
	bool varRptGradAll = false; // also report gradient using all gradients of renew phase
	bool varAvgGradTop = false; // also update average gradient using gradients of the top data points
	bool varVerDP = false; // also update average gradient using gradients of the top data points

	size_t paramWidth; // parameter width

	size_t renewSize;
	size_t renewPointer;
	size_t topSize;

public:
	double stat_t_renew = 0, stat_t_update = 0, stat_t_post = 0;
	double stat_t_u_topk = 0, stat_t_u_grad = 0, stat_t_u_prio = 0, stat_t_u_merge = 0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void prepare(); // after bind data

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);

protected:
	std::vector<double> phaseCalculateGradient(const size_t k);

};
