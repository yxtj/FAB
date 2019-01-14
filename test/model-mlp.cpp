#include <iostream>
#include <string>
#include <vector>
#include "logging/logging.h"
#include "data/DataHolder.h"
#include "model/Model.h"
#include "train/GD.h"
#include "util/Util.h"

using namespace std;

struct Option{
	string fnData;
	vector<int> idSkip;
	vector<int> idY;
	bool withHeader;
	bool doNormalize;
	string shape;
	size_t batchSize;
	double lrate;
	int niter;

	bool parse(int argc, char* argv[]){
		if(argc <= 8)
			return false;
		int idx = 1;
		fnData = argv[idx++];
		idY = getIntList(argv[idx++]);
		withHeader = beTrueOption(argv[idx++]);
		doNormalize = beTrueOption(argv[idx++]);
		shape = argv[idx++];
		batchSize = stoul(argv[idx++]);
		lrate = stod(argv[idx++]);
		niter = stoi(argv[idx++]);
		return true;
	}
	void showUsage(){
		LOG(INFO) << "Usage: <fn-data> <idx-y> <with-header> <normalize> <shape> <batch-size> <lrate> <n-iter>";
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
		LOG(ERROR) << "Cannot parse parameters";
		return 1;
	}

	DataHolder dh(false, 1, 0);
	if(!opt.fnData.empty()){
		dh.load(opt.fnData, ",", {}, opt.idY, opt.withHeader, true);
	} else{
		dh.add({ .2, .9 }, { 0.92 });
		dh.add({ .1, .5 }, { 0.86 });
		dh.add({ .3, .6 }, { 0.89 });

		dh.add({ .9, .2 }, { 0.08 });
		dh.add({ .5, .1 }, { 0.14 });
		dh.add({ .6, .3 }, { 0.11 });
	}
	if(opt.doNormalize)
		dh.normalize(false);
	LOG(INFO) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
	LOG(INFO) << "data[1]: " << dh.get(1).x << " -> " << dh.get(1).y;

	Model m;
	if(!opt.fnData.empty()){
		m.init("mlp", dh.xlength(), opt.shape, 0.01);
	} else{
		m.init("mlp", dh.xlength(), "2-3-1", 0.01);
	}

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
		vector<double> dlt = trainer.batchDelta(p, opt.batchSize, true);
		trainer.applyDelta(dlt);
		double loss = trainer.loss();
		p += opt.batchSize;
		if(p >= dh.size())
			p = 0;
		show(trainer.pm->getParameter().weights, dlt, loss);
		//LOG(INFO) << loss;
	}

	LOG(INFO) << "finish.";
	return 0;
}
