#include "KernelFactory.h"
#include "app/LogisticRegression.h"
#include "app/MLP.h"
#include "app/CNN.h"
#include "app/RNN.h"
#include "app/TopicModel.h"
#include <stdexcept>
using namespace std;

Kernel* KernelFactory::generate(const std::string& name){
	if (name == "lr")
		return new LogisticRegression();
	else if (name == "mlp")
		return new MLP();
	else if (name == "cnn")
		return new CNN();
	else if (name == "rnn")
		return new RNN();
	else if (name == "tm")
		return new TopicModel();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
