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
	size_t batchSize;
	double lrate;
	int niter;

	bool parse(int argc, char* argv[]){
		if(argc <= 8)
			return false;
		fnData = argv[1];
		idSkip = getIntList(argv[2]);
		idY = getIntList(argv[3]);
		withHeader = beTrueOption(argv[4]);
		doNormalize = beTrueOption(argv[5]);
		batchSize = stoul(argv[6]);
		lrate = stod(argv[7]);
		niter = stoi(argv[8]);
		return true;
	}
	void showUsage(){
		LOG(INFO) << "Usage: <fn-data> <idx-skip> <idx-y> <with-header> <normalize> <batch-size> <lrate> <n-iter>";
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

	DataHolder dh(true, 1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, opt.withHeader, true);
	if(opt.doNormalize)
		dh.normalize(false);
	LOG(INFO) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
	LOG(INFO) << "data[1]: " << dh.get(1).x << " -> " << dh.get(1).y;

	Model m;
	m.init("lr", dh.xlength(), to_string(dh.xlength()), 0.01);

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
	}

	LOG(INFO) << "finish.";
	return 0;
}
