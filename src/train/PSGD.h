#pragma once
#include "Trainer.h"
#include "priority/PriorityHolder.h"

class PSGD : public Trainer
{
protected:
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
	PriorityHolder* prhd = nullptr;
	std::vector<float> priority; // buffer for priorities in current iteration
	std::vector<int> priorityIdx; // for top-k
	unsigned wver; // parameter version

	// gradient
	std::vector<double> avgGrad;
	// variations
	bool varAggReport= false; // also report gradient from the priority-update phase
	bool varAggLearn = false; // also calculate and learn the data points from the parameter-update phase
	bool varAggAverage = false; // also update average gradient using gradients from the parameter-update phase
	bool varVerDP = false; // use data points number or iteration as version

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
	virtual void ready(); // after set initializing parameter
	virtual ~PSGD();

	// cond is not used
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);

// main logic:
protected:
	std::vector<double> phaseUpdatePriority(const size_t r);
	std::vector<double> phaseCalculateGradient(const size_t k);

// parse parameters
	bool parsePriority(const std::string& typeInit, const std::string& type);
	bool parseDecay(const std::string& type);
	bool parseVariation(const std::string& str);

// priority
	float calcPriority(const std::vector<double>& g);
	float calcPriorityProjection(const std::vector<double>& g);
	float calcPriorityLength(const std::vector<double>& g);
	using fp_cp_t = decltype(&PSGD::calcPriority);
	fp_cp_t fp_cp;
	void getTopK(const size_t k);
	void moveWver();

// gradient
	void updateAvgGrad(const std::vector<double>& g, const double f);
};
