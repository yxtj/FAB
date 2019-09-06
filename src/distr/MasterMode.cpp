#include "Master.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
using namespace std;

// ---- bulk synchronous parallel

void Master::bspInit()
{
	factorDelta = 1.0 / nWorker;
	if(!trainer->needAveragedDelta())
		factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::bspProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
			tl = t;
		}
		VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
		waitDeltaFromAll();
		stat.t_dlt_wait += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		archiveProgress();
		//waitParameterConfirmed();
		++iter;
	}
}

// ---- typical asynchronous parallel

void Master::tapInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaTap));
}

void Master::tapProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
			newIter = false;
			if(VLOG_IS_ON(2) && iter % ln == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		int p = static_cast<int>(nUpdate / nWorker + 1);
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

// ---- staleness synchronous parallel

void Master::sspInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaSsp));
	deltaIter.assign(nWorker, 0);
	bfDeltaNext.assign(1, vector<double>(trainer->pm->paramWidth(), 0.0));
	bfDeltaDpCountNext.assign(1, 0);
}

void Master::sspProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
			tl = t;
		}
		VLOG_EVERY_N(ln, 2) << "  Waiting for all deltas";
		int p = *min_element(deltaIter.begin(), deltaIter.end());
		while(p < iter){
			VLOG_EVERY_N(ln*nWorker, 2) << "Param-iteration: " << iter << " Delta-iteration: " << deltaIter;
			waitDeltaFromAny();
			p = *min_element(deltaIter.begin(), deltaIter.end());
		}
		// NODE: if is possible that p > iter (moves 2 or more iterations at once)
		//       but we only process one param-iteration in one loop
		stat.t_dlt_wait += tmr.elapseSd();
		{
			lock_guard<mutex> lg(mbfd);
			applyDelta(bfDelta, -1);
			//clearAccumulatedDeltaNext(0);
			shiftAccumulatedDeltaNext();
			++iter;
		}
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		archiveProgress();
	}
}

// ---- staleness asynchronous parallel

void Master::sapInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaSap));
}

void Master::sapProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
			newIter = false;
			if(VLOG_IS_ON(2) && iter % ln == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		int p = static_cast<int>(nUpdate / nWorker + 1);
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

// ---- flexible synchronous parallel

void Master::fspInit()
{
	factorDelta = 1.0 / nWorker;
	if(!trainer->needAveragedDelta())
		factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaFsp));
}

void Master::fspProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
			tl = t;
		}
		VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
		double interval = pie->interval();
		sleep(interval);
		VLOG_EVERY_N(ln, 2) << "  Broadcast pause signal";
		Timer tsync;
		broadcastSignalPause();
		VLOG_EVERY_N(ln, 2) << "  Waiting for all deltas";
		waitDeltaFromAll();
		stat.t_dlt_wait += tmr.elapseSd();
		applyDelta(bfDelta, -1);
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		//waitParameterConfirmed();
		pie->update(bfDelta, interval, bfDeltaDpCount, tsync.elapseSd(), tmrTrain.elapseSd());
		clearAccumulatedDelta();
		//VLOG_EVERY_N(ln, 2) << "  Broadcast continue signal";
		//broadcastSignalContinue();
		archiveProgress();
		++iter;
	}
}

// ---- aggressive asynchronous parallel

void Master::aapInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaAap));
}

void Master::aapProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;// << ", msg waiting: " << driver.queSize() << ", update: " << nUpdate;
			//DVLOG_EVERY_N(ln / 10, 1) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
			newIter = false;
			if(VLOG_IS_ON(2) && iter % ln == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		multicastParameter(lastDeltaSource.load());
		int p = static_cast<int>(nUpdate / nWorker + 1);
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

// ---- progressive asynchronous parallel

void Master::papInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaPap));
	regDSPProcess(MType::DReport, localCBBinder(&Master::handleReport));
}

void Master::papProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	tmrDeltaV.restart();
	if(opt->algorighm == "lda")
		model.resetparam();
	while(!terminateCheck()){
		// VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		VLOG_EVERY_N(20, 1) << "In iteration: " << iter << " update: " << nUpdate
			<< " dp: " << ttDpProcessed;

		if(opt->mode.find("pasp5") != std::string::npos){
			waitDeltaFromAll();
			ttDpProcessed += getDeltaCnt;
			broadcastParameter();
			nUpdate++;
			VLOG_IF(nUpdate < 3, 1) << "pasp5 broadcastParameter: " << iter << " update: " << nUpdate
				<< " dp: " << ttDpProcessed;
			archiveProgressAsync(std::to_string(objImproEsti / getDeltaCnt), false);
			getDeltaCnt = 0;
		} else {
			waitDeltaFromAny();
			suDeltaAny.reset();
		}

		if(unSendDelta == 0){
			++iter;
		}
	}
}


// ---- handlers ----

void Master::handleDelta(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	stat.n_point += deltaMsg.first;
	applyDelta(deltaMsg.second, s);
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info, MType::DDelta);
	++stat.n_dlt_recv;
}

void Master::handleDeltaTap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	stat.n_point += deltaMsg.first;
	applyDelta(deltaMsg.second, s);
	++nUpdate;
	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaSsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	{
		++deltaIter[s];
		lock_guard<mutex> lg(mbfd);
		if(iter == deltaIter[s]){
			accumulateDelta(deltaMsg.second, deltaMsg.first);
		} else{
			accumulateDeltaNext(deltaIter[s] - iter, deltaMsg.second, deltaMsg.first);
		}
	}
	++nUpdate;
	//applyDelta(deltaMsg.second, s); // called at the main process
	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//rph.input(typeDDeltaN, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaSap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	stat.n_point += deltaMsg.first;
	applyDelta(deltaMsg.second, s);
	++nUpdate;
	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaFsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	accumulateDelta(deltaMsg.second, deltaMsg.first);
	//applyDelta(deltaMsg.second, s); // called at the main process
	rph.input(typeDDeltaAll, s);
	//rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaAap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	stat.n_point += deltaMsg.first;
	applyDelta(deltaMsg.second, s);
	++nUpdate;
	//static vector<int> cnt(nWorker, 0);
	//++cnt[s];
	//VLOG_EVERY_N(ln/10, 1) << "Update: " << nUpdate << " rsp: " << cnt << " r-pkg: " << net->stat_recv_pkg;
	//rph.input(typeDDeltaAll, s);
	lastDeltaSource = s;
	rph.input(typeDDeltaAny, s);
	if(conf->aapWait)
		sendReply(info, MType::DDelta);
	++stat.n_dlt_recv;
	// broadcast new parameter in main thread
}

void Master::handleDeltaPap(const std::string& data, const RPCInfo& info)
{
	Timer tmr;
	int s = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	int deltaCnt = delta.back();
	getDeltaCnt += deltaCnt;
	delta.pop_back();

	deltaCount[s] += deltaCnt;
	deltaObj[s] += delta.back();
	objEsti += delta.back();
	delta.pop_back();
	deltaT[s] += tmrDeltaV.elapseSd();
	objImproEsti += delta.back();

	stat.t_data_deserial += tmr.elapseSd();
	applyDelta(delta, s);

	double thread = glbBatchSize;
	if(opt->mode.find("pasp4") != std::string::npos){
		thread = glbBatchSize * shrinkFactor;
	}
	//// pasp5 send out the interval
	if(opt->mode.find("pasp5") != std::string::npos && nUpdate == 0){
		double tt = tmrTrain.elapseSd();
		VLOG(1) << "Broadcast Interval: " << tt << " for " << deltaCnt << " from " << s;
		net->broadcast(MType::CTrainInterval, tt * opt->reptr / deltaCnt / nWorker);
	}

	if(opt->algorighm == "mlp"){
		double objImprove = l1norm0(delta);
		staleStats += "_" + std::to_string(s) + "_" + std::to_string(objImprove);
		objImproEsti += objImprove;
	}

	VLOG(3) << " Rev Delta from " << s << ", " << deltaCnt << ", getDeltaCnt: " << getDeltaCnt
		<< ", thread: " << thread;

	rph.input(typeDDeltaAll, s);

	if(getDeltaCnt >= thread) { // && opt->mode.find("pasp5") == std::string::npos) {
		rph.input(typeDDeltaAny, s);
		// ++unSendDelta;
	// if (unSendDelta >= freqSendParam){
		broadcastParameter();
		sentDReq = false;
		tmrDeltaV.restart();
		deltaV.assign(nWorker + 3, 0);
		ttDpProcessed += getDeltaCnt;
		reportNum = 0;
		fastReady = false;
		factorReady = false;

		++nUpdate;
		// VLOG_IF(nUpdate<5,1) << "pasp4 shrinkFactor: " << shrinkFactor;
		string states = std::to_string(getDeltaCnt) + ";" + std::to_string(objEsti)
			+ ";" + std::to_string(objImproEsti);
		for(int i = 0; i < nWorker; i++){
			states += ";" + std::to_string(deltaCount[i]) + "_" + std::to_string(deltaObj[i]) + "_"
				+ std::to_string(deltaT[i]);
		}

		deltaCount.assign(nWorker, 0);
		deltaT.assign(nWorker, 0.0);
		deltaObj.assign(nWorker, 0.0);

		// archiveProgressAsync(std::to_string(shrinkFactor)+"_"
		// 		+std::to_string(objImproEsti/getDeltaCnt), false);
		archiveProgressAsync(states, false);
		shrinkFactor = 1.0;
		getDeltaCnt = 0;
		objImproEsti = 0.0;
		objEsti = 0.0;
		// }
	}
	//sendReply(info);<< "," << objImproEsti
	++stat.n_dlt_recv;
	// directly send new parameter
	// sendParameter(s);
	// ++unSendDelta;
	// if (unSendDelta >= freqSendParam){
	// 	broadcastParameter();
	// 	++nUpdate;
	// 	unSendDelta = 0;
	// 	archiveProgressAsync("", true);
	// }
}