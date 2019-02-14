#include "KernelFactory.h"
#include "LogisticRegression.h"
#include "MLP.h"
#include "CNN.h"
#include "TopicModel.h"
#include "RNN.h"
#include <stdexcept>
using namespace std;

Kernel* KernelFactory::generate(const std::string& name){
	if (name == "lr")
		return new LogisticRegression();
	else if (name == "mlp")
		return new MLP();
	else if (name == "cnn")
		return new CNN();
	else if (name == "tm")
		return new TopicModel();
	else if (name == "rnn")
		return new RNN();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
