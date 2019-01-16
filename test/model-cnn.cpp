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
		if(fnData == "-")
			fnData.clear();
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
	Model m;
	if(!opt.fnData.empty()){
		dh.load(opt.fnData, ",", {}, opt.idY, opt.withHeader, true);
		m.init("cnn", dh.xlength(), opt.shape, 0.01);
	} else{
		// pattern: have sequence 0.3, 0.9, 0.6
		dh.add({ 0.0, 0.2, 0.8, 0.3, 0.9, 0.6, 0.4, 0.9 }, { 1 });
		dh.add({ 0.3, 0.9, 0.3, 0.6, 0.3, 0.9, 0.6, 0.6 }, { 1 });
		dh.add({ 0.3, 0.9, 0.6, 0.3, 0.9, 0.6, 0.6, 0.1 }, { 1 });
		dh.add({ 0.2, 0.9, 0.6, 0.3, 0.9, 0.6, 0.6, 0.1 }, { 0.9 });
		dh.add({ 0.3, 0.9, 0.3, 0.6, 0.4, 0.9, 0.5, 0.6 }, { 0.8 });


		dh.add({ 0.3, 0.1, 0.9, 0.5, 0.3, 0.6, 0.6, 0.1 }, { 0 });
		dh.add({ 0.7, 0.9, 0.3, 0.3, 0.3, 0.5, 0.6, 0.1 }, { 0 });
		dh.add({ 0.2, 0.3, 0.6, 0.9, 0.9, 0.2, 0.6, 0.1 }, { 0.1 });
		dh.add({ 0.3, 0.7, 0.3, 0.6, 0.4, 0.9, 0.5, 0.6 }, { 0.2 });

		m.init("cnn", dh.xlength(), "8-1,c,3-1,a,relu-1,p,max,2-1,f", 0.01);
	}
	if(opt.doNormalize)
		dh.normalize(false);
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
