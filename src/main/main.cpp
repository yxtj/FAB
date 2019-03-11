#include "logging/logging.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "data/DataHolder.h"
#include "distr/Master.h"
#include "distr/Worker.h"
#include "func.h"
#include <string>
#include <vector>
//#include <iostream>

using namespace std;

int main(int argc, char* argv[]){
	initLogger(argc, argv);
	NetworkThread::Init(argc, argv);
	NetworkThread* net = NetworkThread::GetInstance();
	Option opt;
	if(net->size() == 1){
		LOG(ERROR) << "# of MPI instance should be larger than 1";
		NetworkThread::Shutdown();
		opt.showUsage();
		return 2;
	}
	if(!opt.parse(argc, argv, net->size() - 1)){
		if(net->id() == 0)
			opt.showUsage();
		NetworkThread::Shutdown();
		return 1;
	}
	LOG(DEBUG) << "size=" << net->size() << " id=" << net->id();
	if(net->id() == 0){
		LOG(INFO) << "Infromation:\nData file: " << opt.fnData << " \tNormalize: " << opt.doNormalize
			<< "\n  Idx-y: " << opt.idY << "\tIdx-skip: " << opt.idSkip
			<< "\nAlgorithm: " << opt.algorighm << "\tParam: " << opt.algParam
			<< "\n  Interval Estimator: " << opt.intervalParam << "\tMulticast: " << opt.mcastParam
			<< "\nRecord file: " << opt.fnOutput
			<< "\nTraining configurations:\n  Mode: " << opt.mode << "\tLearning-rate: " << opt.lrate
			<< "\tBatch-size: " << opt.batchSize << "\tWorker-#: " << opt.nw
			<< "\nTerminating condition:\n  Max-iteration: " << opt.tcIter << "\tMax-time: " << opt.tcTime
			<< "\nArchive iteration: " << opt.arvIter << "\tinterval: " << opt.arvTime
			<< "\nLog iteration: " << opt.logIter;
	}
#ifndef NDEBUG
	if(net->id()==0){
		DLOG(DEBUG)<<"pause.";
		DLOG(DEBUG)<<cin.get();
	}
#endif
	if(net->id()==0){
		Master m;
		m.init(&opt, 0);
		m.run();
	} else{
		size_t lid = net->id()-1;
		Worker w;
		w.init(&opt, lid);
		VLOG(2) << "Loading data";
		DataHolder dh(false, opt.nw, lid);
		try{
			dh.load(opt.fnData, ",", opt.idSkip, opt.idY, opt.header, true);
			DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			if(opt.doNormalize)
				dh.normalize(false);
			DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			w.bindDataset(&dh);
			w.run();
		} catch(exception& e){
			LOG(FATAL) << "Error in loading data file: " << opt.fnData << "\n" << e.what() << endl;
		}
	}

	NetworkThread::Shutdown();
	//NetworkThread::Terminate();
	return 0;
}
