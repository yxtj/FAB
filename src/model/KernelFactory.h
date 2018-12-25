#pragma once
#include <string>
#include "Kernel.h"

struct KernelFactory
{
	static Kernel* generate(const std::string& name);
};
