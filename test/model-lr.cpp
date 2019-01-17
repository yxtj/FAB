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
	int showIter;

	bool parse(int argc, char* argv[]){
		int optIdx = 7;
		if(argc <= optIdx)
			return false;
		int idx = 1;
		try{
			fnData = argv[idx++];
			idY = getIntList(argv[idx++]);
			withHeader = beTrueOption(argv[idx++]);
			doNormalize = beTrueOption(argv[idx++]);
			batchSize = stoul(argv[idx++]);
			lrate = stod(argv[idx++]);
			niter = stoi(argv[idx++]);
			showIter = argc <= optIdx++ ? 1 : stoi(argv[idx++]);
		} catch(exception& e){
			LOG(ERROR) << e.what();
			LOG(ERROR) << "cannot parse " << idx << "-th parameter: " << argv[idx];
			return false;
		}
		return true;
	}
	void showUsage(){
		LOG(ERROR) << "Usage: <fn-data> <idx-y> <with-header> <normalize> <batch-size> <lrate> <n-iter>  [iter-show]";
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
		if(opt.showIter != 0 && iter%opt.showIter == 0)
			show(trainer.pm->getParameter().weights, dlt, loss);
		else
			LOG(INFO) << loss;
	}

	LOG(INFO) << "finish.";
	return 0;
}
