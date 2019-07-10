#pragma once
#include "../Trainer.h"
#include "../PSGD.h"
#include "../priority/PriorityHolder.h"
#include "../priority/PriorityDumper.h"

class PSGD_log : public PSGD
{
	PriorityDumper pdump;

public:
	virtual void init(const std::vector<std::string>& param);
	virtual std::string name() const;
	virtual void prepare(); // after bind data

	virtual DeltaResult batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);

};
