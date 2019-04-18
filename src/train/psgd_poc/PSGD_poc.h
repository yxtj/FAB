#pragma once
#include "Trainer.h"
#include <fstream>

class PSGD_poc : public Trainer
{
protected:
	double rate = 1.0;
	double topRatio = 1.0;
	bool global = false; // true->global (gi*avg(g)), false->self (gi*gi)
	size_t paramWidth; // parameter width
	std::string fname;
	std::ofstream fout; // log of gradient
	std::vector<std::vector<double>> gradient;
	std::vector<double> sumGrad;

public:
	double stat_t_grad_calc = 0, stat_t_grad_post = 0;
	double stat_t_grad_archive = 0;
	double stat_t_prio_pick = 0, stat_t_prio_update;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void ready();
	virtual ~PSGD_poc();

protected:
	void updateGradient(const size_t start, const size_t end);
	void dumpGradient();
};
