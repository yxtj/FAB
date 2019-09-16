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
	int showIter;

	bool parse(int argc, char* argv[]){
		int optIdx = 9;
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
			shape = argv[idx++];
			batchSize = stoul(argv[idx++]);
			lrate = stod(argv[idx++]);
			niter = stoi(argv[idx++]);
			showIter = argc <= optIdx++ ? 1 : stoi(argv[idx++]);
		} catch(exception& e){
			LOG(ERROR) << e.what();
			LOG(ERROR) << "cannot parse " << idx-1 << "-th parameter: " << argv[idx-1];
			return false;
		}
		return true;
	}
	void showUsage(){
		LOG(ERROR) << "Usage: <fn-data> <idx-y> <with-header> <normalize> <shape> <batch-size> <lrate> <n-iter> [iter-show]";
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
	Model m;
	if(!opt.fnData.empty()){
		dh.load(opt.fnData, ",", {}, opt.idY, opt.withHeader, true);
		m.init("cnn", opt.shape, 123456u);
	} else{
		dh.setLength(8, 1);
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

		m.init("cnn", "8-1c3-sigmoid-max:2-1f", 123456u);
	}
	if(!m.checkData(dh.xlength(), dh.ylength()))
		LOG(FATAL) << "data size does not match model";
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

	auto g = m.gradient(dh.get(0));
	auto a = m.forward(dh.get(0));
	auto b = m.backward(dh.get(0));
	loss = m.loss(a, dh.get(0).y);
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
