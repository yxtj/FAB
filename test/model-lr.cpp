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
		int optIdx = 8;
		if(argc < optIdx)
			return false;
		int idx = 1;
		try{
			fnData = argv[idx++];
			if(fnData == "-" || fnData == " ")
				fnData.clear();
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

	DataHolder dh(1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, opt.withHeader, true);
	if(opt.doNormalize)
		dh.normalize(false);
	LOG(INFO) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
	LOG(INFO) << "data[1]: " << dh.get(1).x << " -> " << dh.get(1).y;

	Model m;
	m.init("lr", to_string(dh.xlength()), 123456U);
	if(!m.checkData(dh.xlength(), dh.ylength()))
		LOG(FATAL) << "data size does not match model";

	GD trainer;
	trainer.setRate(opt.lrate);
	trainer.bindDataset(&dh);
	trainer.bindModel(&m);

	auto g = m.gradient(dh.get(0));
	auto a = m.forward(dh.get(0));
	auto b = m.backward(dh.get(0));
	auto loss = m.loss(a, dh.get(0).y);
	double diff = 0.0;
	for(size_t i = 0; i < g.size(); ++i){
		auto v = g[i] - b[i];
		diff += v * v;
	}
	LOG(INFO) << diff << "\t" << loss << endl;

	LOG(INFO) << "start";
	show(trainer.pm->getParameter().weights, {}, trainer.loss());
	atomic_bool flag;
	size_t p = 0;
	for(int iter = 0; iter < opt.niter; ++iter){
		LOG(INFO) << "Iteration: " << iter;
		auto dr = trainer.batchDelta(flag, p, opt.batchSize, true);
		trainer.applyDelta(dr.delta);
		double loss = trainer.loss();
		p += opt.batchSize;
		if(p >= dh.size())
			p = 0;
		if(opt.showIter != 0 && iter%opt.showIter == 0)
			show(trainer.pm->getParameter().weights, dr.delta, loss);
		else
			LOG(INFO) << loss;
	}

	LOG(INFO) << "finish.";
	return 0;
}
