#include "logging/logging.h"
#include "network/NetworkThread.h"
#include "data/DataLoader.h"
#include "data/DataHolder.h"
#include "distr/Master.h"
#include "distr/Worker.h"
#include "message/MType.h"
#include "util/Timer.h"
#include "util/Sleeper.h"
#include "Option.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;

string makeSpeedString(const ConfData& conf){
	string tmpSpeed;
	if(conf.adjustSpeedRandom){
		tmpSpeed = "\n  Random: ";
		for(auto& s : conf.speedRandomParam)
			tmpSpeed += s + ",";
		if(!conf.speedRandomParam.empty())
			tmpSpeed.pop_back();
		tmpSpeed += "\tRandom Min: " + to_string(conf.speedRandomMin)
			+ "\tRandom Max: " + to_string(conf.speedRandomMax);
	}
	if(conf.adjustSpeedHetero){
		tmpSpeed += "\n  Heterogenerity: ";
		for(size_t i = 0; i < conf.nw; ++i){
			tmpSpeed += to_string(i) + ":{";
			for(auto& p : conf.speedHeterogenerity[i])
				tmpSpeed += "(" + to_string(p.first) + "," + to_string(p.second) + "),";
			if(!conf.speedHeterogenerity[i].empty())
				tmpSpeed.pop_back();
			tmpSpeed += "}\t";
		}
	}
	return tmpSpeed;
}

int main(int argc, char* argv[]){
	iostream::sync_with_stdio(false);
	initLogger(argc, argv);
	NetworkThread::Init(argc, argv);
	NetworkThread* net = NetworkThread::GetInstance();
	Option opt;
	if(!opt.parse(argc, argv, static_cast<size_t>(net->size()) - 1) || net->size() == 1){
		NetworkThread::Shutdown();
		if(net->id() == 0)
			opt.showUsage();
		return 1;
	}
	Timer::Init();
	Sleeper::Init();
	DLOG(INFO) << "size=" << net->size() << " id=" << net->id()
		<< " overhead-measure=" << Sleeper::GetMeasureOverhead() << " overhead-sleep=" << Sleeper::GetSleepOverhead();
	if(net->id() == 0){
		string tmpSpeed = makeSpeedString(opt.conf);
		LOG(INFO) << "Infromation:"
			// dataset
			<< "\nDataset: " << opt.conf.dataset << "\tLocation: " << opt.conf.fnData
			<< "\n  Normalize: " << opt.conf.normalize << "\tRandom Shuffle: " << opt.conf.shuffle
			<< "\tTrainPart: " << opt.conf.trainPart
			<< "\n  Separator: " << opt.conf.sepper << "\tIdx-y: " << opt.conf.idY << "\tIdx-skip: " << opt.conf.idSkip
			// cluster
			<< "\nCluster: " << "\tWorker-#: " << opt.conf.nw << "\tSpeed random: " << opt.conf.adjustSpeedRandom
			<< "\tSpeed heterogenerity: " << opt.conf.adjustSpeedHetero << tmpSpeed
			// algorithm
			<< "\nAlgorithm: " << opt.conf.algorighm << "\tParam: " << opt.conf.algParam << "\tSeed: " << opt.conf.seed
			<< "\n  Interval Estimator: " << opt.conf.intervalParam << "\tMulticast: " << opt.conf.mcastParam
			// result
			<< "\nRecord file: " << opt.conf.fnOutput << "\tBinary: " << opt.conf.binary
			// mode
			<< "\nTraining configurations:\n  Mode: " << opt.conf.mode
			<< "\tOptimizer: " << opt.conf.optimizer << "\tParam: " << opt.conf.optimizerParam
			<< "\tBatch-size: " << opt.conf.batchSize
			<< "\n  Probe: "<<opt.conf.probe << "\tProbe ratio: " << opt.conf.probeRatio
			<< "\tMin probe GBS: " << opt.conf.probeMinGBSR
			// termination & archive & log
			<< "\nTerminating condition:\n  Max-iteration: " << opt.conf.tcIter << "\tMax-time: " << opt.conf.tcTime
			<< "\nArchive iteration: " << opt.conf.arvIter << "\tinterval: " << opt.conf.arvTime
			<< "\nLog iteration: " << opt.conf.logIter;
	}
#if !defined(NDEBUG) || defined(_DEBUG)
	if(net->id()==0){
		DLOG(DEBUG)<<"pause.";
		DLOG(DEBUG)<<cin.get();
	}
#endif
	if(net->id()==0){
		Master m;
		m.init(&opt.conf, 0);
		m.run();
	} else{
		size_t lid = static_cast<size_t>(net->id()) - 1;
		Worker w;
		w.init(&opt.conf, lid);
		DataHolder dh;
		try{
			DataLoader dl;
			dl.init(opt.conf.dataset, opt.conf.nw, lid, true);
			if(opt.conf.dataset == "csv" || opt.conf.dataset == "tsv" || opt.conf.dataset == "customize")
				dl.bindParameterTable(opt.conf.sepper, opt.conf.idSkip, opt.conf.idY, opt.conf.header);
			else if(opt.conf.dataset == "list")
				dl.bindParameterVarLen(opt.conf.sepper, opt.conf.lenUnit, opt.conf.idY);
			VLOG(1) << "Loading data";
			size_t localk = opt.conf.topk / opt.conf.nw + (lid < opt.conf.topk%opt.conf.nw ? 1 : 0);
			dh = dl.load(opt.conf.fnData, opt.conf.trainPart, localk);
			DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			if(opt.conf.normalize){
				dh.normalize(false);
				DVLOG(2) << "data[0]: " << dh.get(0).x << " -> " << dh.get(0).y;
			}
			if(opt.conf.shuffle){
				VLOG(1) << "Shuffle data";
				dh.shuffle();
			}
		} catch(exception& e){
			LOG(FATAL) << "Error in loading data file: " << opt.conf.fnData << "\n" << e.what() << endl;
		}
		w.bindDataset(&dh);
		w.run();
	}

	net->cancel({ CType::Data });
	NetworkThread::Shutdown();
	//NetworkThread::Terminate();
	return 0;
}
