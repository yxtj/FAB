#pragma once
#include "Trainer.h"

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
	//double prioDecayFactor = -1;
	// gradient
	//enum struct GradientType{
	//	Increment, // keep the summation incrementally
	//	Decay, // use the decay implementation for exponential average
	//};
	//float gradDecayFactor = 0.9;
	// variations
	bool varUpdateRptGradAll = false; // also report gradient using ALL gradients of renew phase
	bool varUpdateRptGradSel = false; // also report gradient using SOME gradients of renew phase (priority>=threshold)
	bool varUpdateAvgGradTop = false; // also update average gradient using gradients of the top data points

	size_t paramWidth; // parameter width
	std::vector<double> avgGrad;
	std::vector<float> priority;
	std::vector<int> priorityIdx; // for top-k
	float prioThreshold; // for variation-RptGradSel

	//std::vector<float> priorityOld; // for priority decay
	std::vector<float> priorityDecayRate; // for priority decay
	std::vector<unsigned> priorityWver; // for priority decay
	unsigned wver; // for priority decay

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
	void updatePriority(float p, size_t id);
	void updatePriorityKeep(float p, size_t id);
	void updatePriorityDecay(float p, size_t id);
	using fp_up_t = decltype(&PSGD::updatePriority);
	fp_up_t fp_up;
	void getTopK(const size_t k);
	bool topKPredKeep(const int l, const int r);
	bool topKPredDecay(const int l, const int r);
	using fp_pkp_t = decltype(&PSGD::topKPredKeep);
	fp_pkp_t fp_pkp;
// gradient
private:
	void updateAvgGrad(const std::vector<double>& g, const double f);
};
