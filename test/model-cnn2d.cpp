#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "logging/logging.h"
#include "data/DataHolder.h"
#include "model/Model.h"
#include "train/GD.h"
#include "util/Util.h"

using namespace std;

vector<double> str2list(const string& str,
	const int blength = 0, const double scale = 1.0)
{
	vector<string> line = getStringList(str, ";");
	vector<double> res;
	for(size_t i = 0; i < line.size(); ++i){
		vector<double> t;
		if(blength == 0){
			t = getDoubleList(line[i], " ,");
		} else{
			for(size_t j = 0; j < line.size(); j += blength){
				string unit = line[i].substr(j, blength);
				t.push_back(stod(unit));
			}
		}
		if(scale != 1.0)
			for(auto& v : t)
				v *= scale;
		res.insert(res.end(), t.begin(), t.end());
	}
	return res;
}
vector<double> rndlist(mt19937& gen, const int n, const int m, const double lb, const double up){
	uniform_real_distribution<double> dist(lb,up);
	vector<double> res;
	for(int i = 0; i < n; ++i)
		for(int j = 0; j < m; ++j)
			res.push_back(dist(gen));
	return res;
}

struct Option{
	size_t batchSize;
	double lrate;
	int niter;
	int showIter;

	bool parse(int argc, char* argv[]){
		int optIdx = 4;
		if(argc < optIdx)
			return false;
		int idx = 1;
		try{
			batchSize = stoul(argv[idx++]);
			lrate = stod(argv[idx++]);
			niter = stoi(argv[idx++]);
			showIter = argc <= optIdx++ ? 1 : stoi(argv[idx++]);
		} catch(exception& e){
			LOG(ERROR) << e.what();
			LOG(ERROR) << "cannot parse " << idx - 1 << "-th parameter: " << argv[idx - 1];
			return false;
		}
		return true;
	}
	void showUsage(){
		LOG(ERROR) << "Usage: <batch-size> <lrate> <n-iter> [iter-show]";
	}
};

void show(const vector<double>& weight, const vector<double>& dlt, const double loss){
	LOG(INFO) << "  weight: " << weight;
	LOG(INFO) << "  delta: " << dlt;
	LOG(INFO) << "  cost: " << loss;
}

int main(int argc, char* argv[]){
	initLogger(argc, argv);
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.showUsage();
		return 1;
	}

	mt19937 gen(1);
	DataHolder dh(false, 1, 0);
	auto fun = [](const string& s){return str2list(s, 1, 1.0 / 5); };
	auto foo = [&](){ return rndlist(gen, 4, 4, 0.0, 5.0); };
	// pattern: 505;050;505
	dh.add(fun("5050;0500;5050;0000"), { 1 });
	dh.add(fun("0505;0050;0505;0000"), { 1 });
	dh.add(fun("0000;5050;0500;5050"), { 1 });
	dh.add(fun("0000;0505;0050;0505"), { 1 });
	// noise
	dh.add(fun("0000;5050;0500;5550"), { 1 });
	dh.add(fun("0505;0055;0505;5005"), { 1 });
	dh.add(fun("5050;5500;5050;0005"), { 1 });
	dh.add(fun("5150;0500;5050;0500"), { 1 });
	// rotate
	dh.add(fun("4140;1510;4140;0000"), { 1 });

	dh.add(fun("5040;0000;5050;0000"), { 0 });
	dh.add(fun("5050;5050;5050;0500"), { 0 });
	dh.add(fun("0000;0000;0000;0000"), { 0 });
	dh.add(fun("1040;0110;5050;0055"), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });
	dh.add(foo(), { 0 });

	Model m;
	try{
		//m.init("cnn", "4*4-2c2*2p2*2-1c2*2p2*2-1f", 0.01);
		m.init("cnn", "4*4-2cp2*2-1cp2*2-1f", 123456u);
	} catch(exception& e){
		LOG(FATAL) << e.what();
	}
	if(!m.checkData(dh.xlength(), dh.ylength()))
		LOG(FATAL) << "data size does not match model";

	//dh.normalize(false);
	LOG(INFO) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
	LOG(INFO) << "data[1]: " << dh.get(1).x << " -> " << dh.get(1).y;

	vector<double> pred = m.predict(dh.get(0));
	double loss = m.loss(pred, dh.get(0).y);
	LOG(INFO) << "pred: " << pred << ", loss: " << loss;

	GD trainer;
	trainer.setRate(opt.lrate);
	trainer.bindDataset(&dh);
	trainer.bindModel(&m);

	LOG(INFO) << "start";
	show(trainer.pm->getParameter().weights, {}, trainer.loss());
	size_t p = 0;
	for(int iter = 0; iter < opt.niter; ++iter){
		LOG(INFO) << "Iteration: " << iter;
		size_t cnt;
		vector<double> dlt;
		tie(cnt, dlt) = trainer.batchDelta(p, opt.batchSize, true);
		trainer.applyDelta(dlt);
		double loss = trainer.loss();
		p += opt.batchSize;
		if(p >= dh.size())
			p = 0;
		if(opt.showIter != 0 && iter%opt.showIter == 0)
			show(trainer.pm->getParameter().weights, dlt, loss);
		else
			LOG(INFO) << loss;
	}

	LOG(INFO) << "finish.";
	return 0;
}
