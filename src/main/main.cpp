#include "logging/logging.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "data/DataLoader.h"
#include "data/DataHolder.h"
#include "distr/Master.h"
#include "distr/Worker.h"
#include "message/MType.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]){
	initLogger(argc, argv);
	NetworkThread::Init(argc, argv);
	NetworkThread* net = NetworkThread::GetInstance();
	Option opt;
	if(!opt.parse(argc, argv, net->size() - 1) || net->size() == 1){
		NetworkThread::Shutdown();
		if(net->id() == 0)
			opt.showUsage();
		return 1;
	}
	DLOG(INFO) << "size=" << net->size() << " id=" << net->id();
	if(net->id() == 0){
		LOG(INFO) << "Infromation:\nDataset: " << opt.dataset << "\tData file: " << opt.fnData
			<< " \tNormalize: " << opt.normalize << "\tRandom Shuffle: " << opt.shuffle
			<< "\n  Separator: " << opt.sepper << "\tIdx-y: " << opt.idY << "\tIdx-skip: " << opt.idSkip
			<< "\nAlgorithm: " << opt.algorighm << "\tParam: " << opt.algParam << "\tSeed: " << opt.seed
			<< "\n  Interval Estimator: " << opt.intervalParam << "\tMulticast: " << opt.mcastParam
			<< "\nRecord file: " << opt.fnOutput << "\tBinary: " << opt.binary
			<< "\nTraining configurations:\n  Mode: " << opt.mode
			<< "\tOptimizer: " << opt.optimizer << "\tParam: " << opt.optimizerParam
			<< "\tBatch-size: " << opt.batchSize << "\tWorker-#: " << opt.nw
			<< "\nTerminating condition:\n  Max-iteration: " << opt.tcIter << "\tMax-time: " << opt.tcTime
			<< "\nArchive iteration: " << opt.arvIter << "\tinterval: " << opt.arvTime
			<< "\nLog iteration: " << opt.logIter;
	}
#if !defined(NDEBUG) || defined(_DEBUG)
	if(net->id()==0){
		DLOG(DEBUG)<<"pause.";
		DLOG(DEBUG)<<cin.get();
	}
#endif
	iostream::sync_with_stdio(false);
	if(net->id()==0){
		Master m;
		m.init(&opt, 0);
		m.run();
	} else{
		size_t lid = net->id()-1;
		Worker w;
		w.init(&opt, lid);
		DataHolder dh;
		try{
			DataLoader dl;
			dl.init(opt.dataset, opt.nw, lid, true);
			dl.bindParameter(opt.sepper, opt.idSkip, opt.idY, opt.header);
			VLOG(1) << "Loading data";
			size_t localk = opt.topk / opt.nw + (lid < opt.topk%opt.nw ? 1 : 0);
			dh = dl.load(opt.fnData, opt.trainPart, localk);
			DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			if(opt.normalize){
				dh.normalize(false);
				DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			}
			VLOG(1) << "Shuffle data";
			if(opt.shuffle){
				dh.shuffle();
			}
		} catch(exception& e){
			LOG(FATAL) << "Error in loading data file: " << opt.fnData << "\n" << e.what() << endl;
		}
		w.bindDataset(&dh);
		w.run();
	}

	net->cancel({ CType::Data });
	NetworkThread::Shutdown();
	//NetworkThread::Terminate();
	return 0;
}
