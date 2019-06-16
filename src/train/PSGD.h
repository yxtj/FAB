#pragma once
#include "Trainer.h"
#include "psgd/PriorityHolder.h"

class PSGD : public Trainer
{
	// parameters
	double rate = 1.0;
	double topRatio = 1.0;
	double renewRatio = 0.01;
	// priority
	enum struct PriorityType{
		Projection, // pi=gi*avg(g)
		Length, // pi=gi*gi
		DecayExp, // pi=gi*avg(g)*exp(a*t)
	};
	PriorityType prioType = PriorityType::Projection;
	PriorityType prioInitType = PriorityType::Length;
//	std::vector<float> priority;
	PriorityHolderExpTwice prhd;
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
	virtual void prepare();
	virtual void ready();
	virtual ~PSGD();

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual DeltaResult batchDelta(std::atomic<bool>& cond,
		const size_t start, const size_t cnt, const bool avg = true);

// main logic:
private:
	std::pair<size_t, std::vector<double>> phaseUpdatePriority(const size_t r);
	std::pair<size_t, std::vector<double>> phaseCalculateGradient(const size_t k);

// parse parameters
private:
	bool parsePriority(const std::string& typeInit, const std::string& type);
	bool parseVariation(const std::string& str);
// priority
private:
	float calcPriority(const std::vector<double>& g);
	float calcPriorityProjection(const std::vector<double>& g);
	float calcPriorityLength(const std::vector<double>& g);
	using fp_cp_t = decltype(&PSGD::calcPriority);
	fp_cp_t fp_cp;
	void getTopK(const size_t k);
// gradient
private:
	void updateAvgGrad(const std::vector<double>& g, const double f);
};
