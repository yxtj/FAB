#pragma once
#include "Trainer.h"
#include <vector>
#include <string>

struct TrainerFactory{
	static Trainer* generate(
		const std::string& name, const std::vector<std::string>& param);
};
