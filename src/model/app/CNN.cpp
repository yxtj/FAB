#include "CNN.h"
#include "util/Util.h"
#include <stdexcept>
#include <regex>
using namespace std;

// -------- CNN --------

void CNN::init(const std::string & param)
{
	initBasic(param);
    // example: 10x10,4c3x3,relu,max2x2,f
	// example: 10-4:c:3-1:relu-1:max:2-1:f
    // format: <n>:<type>[:<shape>]
    //     shape of convolutional node: <k1>*<k2>
    //     shape of fully-connected node: none
    try{
		string pm = preprocessParam(param);
		net.init(pm);
		net.bindGradLossFunc(&CNN::gradLoss);
    }catch(exception& e){
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

bool CNN::checkData(const size_t nx, const size_t ny)
{
	// check input layer size
	if(nx != net.lenFeatureLayer[0])
		throw invalid_argument("The dataset does not match the input layer of the network");
	// check output layer size
	if(ny != 0 && ny != net.lenFeatureLayer.back()*net.numFeatureLayer.back())
		throw invalid_argument("The dataset does not match the output layer of the network");
	return true;
}

std::string CNN::name() const{
	return "cnn";
}

int CNN::lengthParameter() const
{
	return net.lengthParameter();
}

std::vector<double> CNN::predict(
	const std::vector<std::vector<double>>& x, const std::vector<double>& w) const
{
	return net.predict(x[0], w);
}

int CNN::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double CNN::loss(const std::vector<double>& pred, const std::vector<double>& label) const {
	double res = 0.0;
	for(size_t i = 0; i < pred.size(); ++i){
		double t = pred[i] - label[i];
		res += t * t;
	}
	//return 0.5 * res;
	return res;
}

std::vector<double> CNN::gradLoss(const std::vector<double>& pred, const std::vector<double>& label)
{
	vector<double> res(pred.size());
	for(size_t i = 0; i < pred.size(); ++i){
		res[i] = pred[i] - label[i];
	}
	return res;
}

std::vector<double> CNN::gradient(const std::vector<std::vector<double>>& x,
	const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph) const
{
	return net.gradient(x[0], w, y);
}

std::string CNN::preprocessParam(const std::string & param)
{
	string res = procUnitCPx(param);
	res = procUnitCx(res);
	res = procUnitPx(res);
	return res;
}

std::string CNN::procUnitCPx(const std::string & param)
{
	string srShape = net.getRegShape();
	string srSep = "(?:[" + net.getRawSep() + "]?)";
	// -3cp4 -> -3:c:4-sig-max:4
	regex runit_cp(srSep + "(\\d+)?c([srt])?p:?" + srShape);

	string res;
	string buf = param;
	auto first = buf.cbegin();
	auto last = buf.cend();
	std::smatch sm;
	// 3c5 ->3:c:5-sig
	while(regex_search(first, last, sm, runit_cp)) {
		first = sm[0].second; // the last matched position
		string cnt = "1";
		if(sm[1].matched)
			cnt = sm[1].str();
		string act;
		if(!sm[2].matched || sm[2] == "s"){
			act = "sigmoid";
		} else if(sm[3] == "r"){
			act = "relu";
		} else if(sm[3] == "t"){
			act = "tanh";
		} else{
			act = sm[2].str();
		}
		res += sm.prefix().str() + "-" + cnt + ":c:" + sm[3].str() + "-"
			+ act + "-max:" + sm[3].str();
	}
	res += string(first, last);
	return res;
}

std::string CNN::procUnitCx(const std::string & param)
{
	string srShape = net.getRegShape();
	string srSep = "(?:[" + net.getRawSep() + "]?)";
	// 3c5p4 -> 3:c:5-sig-max:4
	regex runit_c(srSep + "(\\d+)c" + srShape + "([srt])?");

	string res;
	string buf = param;
	auto first = buf.cbegin();
	auto last = buf.cend();
	std::smatch sm;
	// 3c5 ->3:c:5-sig
	while(regex_search(first, last, sm, runit_c)) {
		first = sm[0].second; // the last matched position
		string cnt = "1";
		if(sm[1].matched)
			cnt = sm[1].str();
		string act;
		if(!sm[3].matched || sm[3] == "s"){
			act = "sigmoid";
		} else if(sm[3] == "r"){
			act = "relu";
		} else if(sm[3] == "t"){
			act = "tanh";
		} else{
			act = sm[3].str();
		}
		res += sm.prefix().str() + "-" + cnt + ":c:" + sm[2].str() + "-" + act;
	}
	res += string(first, last);
	return res;
}

std::string CNN::procUnitPx(const std::string & param)
{
	string srShape = net.getRegShape();
	string srSep = net.getRegSep() + "?";
	// -p3 -> -max:3
	// p3 -> -max:3
	regex runit_p(srSep + "(\\d+)?p" + srShape); // the sep is important (eg. 3c5p3p2)

	string res;
	string buf = param;
	auto first = buf.cbegin();
	auto last = buf.cend();
	std::smatch sm;
	while(regex_search(first, last, sm, runit_p)) {
		first = sm[0].second; // the last matched position
		string n = "1";
		if(sm[1].matched)
			n = sm[1].str();
		res += sm.prefix().str() + "-" + n + ":max:" + sm[2].str();
	}
	res += string(first, last);

	return res;
}

