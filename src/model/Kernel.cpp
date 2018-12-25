#include "Kernel.h"
using namespace std;

void Kernel::initBasic(const int xlength, const std::string& param){
	this->xlength = xlength;
	this->param = param;
}


std::string Kernel::parameter() const{
	return param;
}
