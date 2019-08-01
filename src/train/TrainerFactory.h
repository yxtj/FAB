#pragma once
#include "Trainer.h"
#include <vector>
#include <string>

struct TrainerFactory{
	static std::vector<std::string> supportList();
	static bool isSupported(const std::string& name);

	static Trainer* generate(
		const std::string& name, const std::vector<std::string>& param);
};
