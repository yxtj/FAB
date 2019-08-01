#include "KernelFactory.h"
#include "app/LogisticRegression.h"
#include "app/MLP.h"
#include "app/CNN.h"
#include "app/RNN.h"
#include "app/TopicModel.h"
#include "app/KMeans.h"
#include <stdexcept>
#include <algorithm>

using namespace std;

std::vector<std::string> KernelFactory::supportList()
{
	static vector<string> supported = { "lr", "mlp", "cnn", "rnn", "km" }; //, "tm"
	return supported;
}

bool KernelFactory::isSupported(const std::string& name)
{
	const auto&& supported = supportList();
	auto it = find(supported.begin(), supported.end(), name);
	return it != supported.end();
}

Kernel* KernelFactory::generate(const std::string& name){
	if(name == "lr")
		return new LogisticRegression();
	else if(name == "mlp")
		return new MLP();
	else if(name == "cnn")
		return new CNN();
	else if(name == "rnn")
		return new RNN();
	else if(name == "tm")
		return new TopicModel();
	else if(name == "km")
		return new KMeans();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
