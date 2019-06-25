#include "Kernel.h"
using namespace std;

void Kernel::initBasic(const std::string& param){
	this->param = param;
}


std::string Kernel::parameter() const{
	return param;
}

bool Kernel::needInitParameterByData() const{
	return false;
}

int Kernel::lengthHidden() const{
	return 0;
}

void Kernel::initVariables(const std::vector<std::vector<double>>& x,
	std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph)
{
}
