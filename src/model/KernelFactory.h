#pragma once
#include "Kernel.h"
#include <string>
#include <vector>

struct KernelFactory
{
	static std::vector<std::string> supportList();
	static bool isSupported(const std::string& name);

	static Kernel* generate(const std::string& name);
};
