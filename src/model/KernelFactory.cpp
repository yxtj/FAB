#include "KernelFactory.h"
#include "LogisticRegression.h"
#include "MLP.h"
#include <stdexcept>
using namespace std;

Kernel* KernelFactory::generate(const std::string& name){
	if(name == "lr")
		return new LogisticRegression();
	else if(name == "mlp")
		return new MLP();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
