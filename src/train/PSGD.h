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

public:
	double stat_t_grad_renew = 0, stat_t_grad_calc = 0, stat_t_grad_post = 0;
	double stat_t_prio_pick = 0, stat_t_prio_update = 0;

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

// parse parameters
private:
	bool parsePriority(const std::string& typeInit, const std::string& type);
	bool parseGradient(const std::string& type, const std::string& factor);
	bool parseVariation(const std::string& str);
// priority
private:
	float calcPriority(const std::vector<double>& g);
	float calcPriorityProjection(const std::vector<double>& g);
	float calcPriorityLength(const std::vector<double>& g);
	using fp_cp_t = decltype(&PSGD::calcPriority);
	fp_cp_t fp_cp;
// gradient
private:
	void updateAvgGrad(const std::vector<double>& g, const double f);
// main logic:
private:
	std::pair<size_t, std::vector<double>> phaseUpdatePriority(const size_t r);
	std::pair<size_t, std::vector<double>> phaseCalculateGradient(const size_t k);
	void getTopK(const size_t k);
	void getTopKDecay(const size_t k);
	void updatePriorityDecay(float p, size_t id);
};
