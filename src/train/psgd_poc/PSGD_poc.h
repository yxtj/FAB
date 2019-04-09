#pragma once
#include "Trainer.h"
#include <fstream>

class PSGD_poc : public Trainer
{
protected:
	double rate = 1.0;
	double topRatio = 1.0;
	size_t paramWidth; // parameter width
	std::string fname;
	std::ofstream fout; // log of gradient
	std::vector<std::vector<double>> gradient;

public:
	double stat_t_grad_calc = 0, stat_t_grad_update = 0, stat_t_grad_archive = 0;
	double stat_t_priority = 0;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void ready();
	virtual ~PSGD_poc();

protected:
	std::pair<size_t, std::vector<double>> batchDelta_point(
		const size_t start, const size_t cnt, const bool avg);
	std::pair<size_t, std::vector<double>> batchDelta_dim(
		const size_t start, const size_t cnt, const bool avg);

	void updateGradient(const size_t start, const size_t end);
	void dumpGradient();
};
