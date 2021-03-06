#include "RNN.h"
#include "util/Util.h"
#include <regex>
#include <stdexcept>
using namespace std;

// -------- RNN --------

void RNN::init(const std::string & param)
{
	initBasic(param);
	// example: 10*10-4,c,3*3-1,a,relu-1,p,max,2*2-1,f
	// example: 10-4,c,3-1,a,relu-1,p,max,2-1,f
	// format: <n>,<type>[,<shape>]
	//     shape of convolutional node: <k1>*<k2>
	//     shape of fully-connected node: none
	try{
		string pm = preprocessParam(param);
		net.init(pm);
		net.bindGradLossFunc(&RNN::gradLoss);
	} catch(exception& e){
		throw invalid_argument(string("Unable to create network: ") + e.what());
	}
	// check input layer
	if(net.typeLayer[0] != NodeType::Input)
		throw invalid_argument("There is no input layer.");
	// check FC layer
	for(size_t i = 0; i < net.nLayer; ++i){
		if(i != net.nLayer - 1 && net.typeLayer[i] == NodeType::FC){
			throw invalid_argument("Only the last layer can be a FC layer.");
		} else if(i == net.nLayer - 1 && net.typeLayer[i] != NodeType::FC){
			throw invalid_argument("The last layer must be a FC layer.");
		}
	}
}

bool RNN::checkData(const size_t nx, const size_t ny)
{
	// check input layer size
	if(nx != net.lenFeatureLayer[0])
		throw invalid_argument("The dataset does not match the input layer of the network");
	// check output layer size
	if(ny != 0 && ny != net.lenFeatureLayer.back()*net.numFeatureLayer.back())
		throw invalid_argument("The dataset does not match the output layer of the network");
	return true;
}

std::string RNN::name() const{
	return "rnn";
}

int RNN::lengthParameter() const
{
	return net.lengthParameter();
}

std::vector<double> RNN::predict(
	const std::vector<std::vector<double>>& x, const std::vector<double>& w) const
{
	vector<double> res;
	for(auto& line : x)
		res = net.predict(line, w);
	return res;
}

int RNN::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double RNN::loss(const std::vector<double>& pred, const std::vector<double>& label) const {
	double res = 0.0;
	for(size_t i = 0; i < pred.size(); ++i){
		double t = pred[i] - label[i];
		res += t * t;
	}
	//return 0.5 * res;
	return res;
}

std::vector<double> RNN::forward(
	const std::vector<std::vector<double>>& x, const std::vector<double>& w)
{
	vector<double> res;
	for(auto& line : x)
		res = net.forward(line, w);
	return res;
}

std::vector<double> RNN::backward(const std::vector<std::vector<double>>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph)
{
	vector<double> res(net.lenFeatureLayer[0], 0.0);
	for(auto& line : x){
		auto temp = net.backward(line, w, y);
		for(size_t i = 0; i < temp.size(); ++i)
			res[i] += temp[i];
	}
	return res;
}

std::vector<double> RNN::gradLoss(const std::vector<double>& pred, const std::vector<double>& label)
{
	vector<double> res(pred.size());
	for(size_t i = 0; i < pred.size(); ++i){
		res[i] = pred[i] - label[i];
	}
	return res;
}

std::vector<double> RNN::gradient(const std::vector<std::vector<double>>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const
{
	vector<double> res(net.lenFeatureLayer[0], 0.0);
	for(auto& line : x){
		auto temp = net.gradient(line, w, y);
		for(size_t i = 0; i < temp.size(); ++i)
			res[i] += temp[i];
	}
	return res;
}

std::string RNN::preprocessParam(const std::string & param)
{
	string srShape = R"((\d+(?:[\*x]\d+)*))"; // v1[*v2[*v3[*v4]]], "*" can also be "x"
	regex rr(R"((\d+):?r:?)" + srShape); // recurrent layer, 4r5 -> 4rs5
	// add default activation type for recurrent layer
	string res = regex_replace(param, rr, "$1:r:s:$2");
	return res;
}
