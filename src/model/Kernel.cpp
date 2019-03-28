#include "Kernel.h"
using namespace std;

void Kernel::initBasic(const std::string& param){
	this->param = param;
}


std::string Kernel::parameter() const{
	return param;
}
