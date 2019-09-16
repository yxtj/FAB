#include "RandomGenerator.h"
#include <random>
#include <algorithm>
#include <functional>

using namespace std;

struct RandomGenerator::Impl{
	std::mt19937 gen;
	std::exponential_distribution<double> distExp;
	std::normal_distribution<double> distNorm;
	std::uniform_real_distribution<double> distUni;

	double offset;
	std::function<double()> fun;
	void init(const std::vector<std::string>& param,
		const double offset, const unsigned seed);
};

void RandomGenerator::Impl::init(const std::vector<std::string>& param,
	const double offset, const unsigned seed)
{
	this->offset = offset;
	gen.seed(seed);
	if(param.empty() || param[0].empty() || param[0] == "none"){
		fun = [&](){ return this->offset; };
	} else if(param[0] == "exp"){
		distExp = exponential_distribution<double>(stod(param[1]));
		fun = [&](){ return this->offset + distExp(gen); };
	} else if(param[0] == "norm"){
		distNorm = normal_distribution<double>(stod(param[1]), stod(param[2]));
		fun = [&](){ return this->offset + distNorm(gen); };
	} else if(param[0] == "uni"){
		distUni = uniform_real_distribution<double>(stod(param[1]), stod(param[2]));
		fun = [&](){ return this->offset + distUni(gen); };
	}
}

std::vector<std::string> RandomGenerator::supportList()
{
	return { "none", "exp", "norm", "uni" };
}

std::string RandomGenerator::usage()
{
	return "none. Always return 0.\n"
		"exp:<lambda>. Exponential distribution.\n"
		"norm:<mean>,<std>. Normal distribution.\n"
		"uni:<a>,<b>. Uniform distribution.\n";
}

bool RandomGenerator::isSupported(const std::string& name)
{
	const auto&& supported = supportList();
	return find(supported.begin(), supported.end(), name) != supported.end();
}

void RandomGenerator::init(const std::vector<std::string>& param,
	const double offset, const unsigned seed)
{
	if(impl == nullptr)
		impl = new RandomGenerator::Impl();
	impl->init(param, offset, seed);
}

double RandomGenerator::generate()
{
	return impl->fun();
}
