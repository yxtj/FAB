#include "Master.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "math/accumulate.h"
using namespace std;

// ---- general probe mode

void Master::probeModeInit()
{
	(this->*initFun)();
	probeReached = false;
	probeNeededPoint = static_cast<size_t>(conf->probeRatio * nPointTotal);
	lossOnline = 0.0;
	suLoss.reset();
}

void Master::probeModeProcess()
{
	Parameter p0 = model.getParameter();
	suLoss.wait_n_reset();
	double l0 = lossOnline;
	// TODO
	vector<size_t> gbsList = { conf->batchSize, conf->batchSize / 2, conf->batchSize / 4 };
	for(size_t gbs : gbsList){
		probeNeededIter = probeNeededPoint / gbs;
		probeNeededDelta = probeNeededPoint / gbs * nWorker;
		probeReached = false;
		broadcastSizeConf(gbs, 0);
		setTerminateCondition(0, probeNeededPoint, probeNeededDelta, probeNeededIter);
		Timer tmr;
		(this->*processFun)();
		double time = tmr.elapseSd();
		gatherLoss();
		double rate = lossOnline / time;

		gbs /= 2;
	}
}

void Master::handleDeltaProbe(const std::string& data, const RPCInfo& info)
{
	(this->*deltaFun)(data, info);
	++nDelta;
	if(nDelta >= probeNeededDelta){
		probeReached = true;
		
	}
}

// ---- bulk synchronous parallel

void Master::bspInit()
{
	factorDelta = 1.0 / nWorker;
	if(!trainer->needAveragedDelta())
		factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaBsp));
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
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " #-delta: " << nDelta;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		int p = static_cast<int>(nDelta / nWorker + 1);
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
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " #-delta: " << nDelta;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		int p = static_cast<int>(nDelta / nWorker + 1);
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
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;// << ", msg waiting: " << driver.queSize() << ", #-delta: " << nDelta;
			//DVLOG_EVERY_N(ln / 10, 1) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
			newIter = false;
			if(VLOG_IS_ON(2) && iter % ln == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " #-delta: " << nDelta;
		waitDeltaFromAny();
		stat.t_dlt_wait += tmr.elapseSd();
		multicastParameter(lastDeltaSource.load());
		int p = static_cast<int>(nDelta / nWorker + 1);
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
	reportProcEach.assign(nWorker, 0);
	reportProcTotal = 0;
	//if(conf->papSearchBatchSize || conf->papSearchReportFreq){
	wtDatapoint.assign(nWorker, 0.0);
	wtDelta.assign(nWorker, 0.0);
	wtReport.assign(nWorker, 0.0);
	//}
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaPap));
	regDSPProcess(MType::DReport, localCBBinder(&Master::handleReport));
}

void Master::papProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;// << ", msg waiting: " << driver.queSize() << ", #-delta: " << nDelta;
		//DVLOG_EVERY_N(ln / 10, 1) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
			tl = t;
			double mtu = mtDeltaSum / nDelta;
			double mtb = mtParameterSum / stat.n_par_send;
			double mtr = mtReportSum / nReport;
			double mto = mtOther / iter;

			double wtd = hmean(wtDatapoint);
			double wtc = mean(wtDelta);
			double wtr = mean(wtReport);

			VLOG(2) << "mtu=" << mtu << "\tmtb=" << mtb << "\tmtr=" << mtr << "\tmto=" << mtOther
				<< "\twtd=" << wtd << "\twtc=" << wtc << "\twtr=" << wtr << "\tloss=" << lossOnline;
		}
		mtOther += tmr.elapseSd();
		// wait until the report counts reach a global mini batch
		suPap.wait_n_reset();
		//// online change globalBatchSize
		if(conf->papDynamicBatchSize) {
			globalBatchSize = estimateGlobalBatchSize();
			VLOG(2) << "gbs=" << globalBatchSize << "\tlrs=" << localreportSize;
		}
		gatherDelta();
		stat.t_dlt_wait += tmr.elapseSd();
		broadcastParameter();

		tmr.restart();
		archiveProgress();
		++iter;
		mtOther += tmr.elapseSd();
	}
}

// ---- handlers ----

void Master::handleDeltaBsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;

	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info, MType::DDelta);
}

void Master::handleDeltaTap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;

	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaSsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	size_t n = get<0>(deltaMsg);
	{
		++deltaIter[s];
		lock_guard<mutex> lg(mbfd);
		// applied in the main process
		if(iter == deltaIter[s]){
			accumulateDelta(get<1>(deltaMsg), n);
		} else{
			accumulateDeltaNext(deltaIter[s] - iter, get<1>(deltaMsg), n);
		}
	}
	updateOnlineLoss(get<2>(deltaMsg), s);
	nPoint += n;
	++nDelta;
	//applyDelta(deltaMsg.second, s); // called at the main process
	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//rph.input(typeDDeltaN, s);
	//sendReply(info);
}

void Master::handleDeltaSap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;

	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaFsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	int s = wm.nid2lid(info.source);
	size_t n = get<0>(deltaMsg);
	nPoint += n;
	//applyDelta(get<1>(deltaMsg), s);
	accumulateDelta(get<1>(deltaMsg), n); // applied in the main process
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;

	rph.input(typeDDeltaAll, s);
	//rph.input(typeDDeltaAny, s);
	//sendReply(info);
}

void Master::handleDeltaAap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;
	//static vector<int> cnt(nWorker, 0);
	//++cnt[s];
	//VLOG_EVERY_N(ln/10, 1) << "#-delta: " << nDelta << " rsp: " << cnt << " r-pkg: " << net->stat_recv_pkg;
	//rph.input(typeDDeltaAll, s);
	lastDeltaSource = s;
	rph.input(typeDDeltaAny, s);
	if(conf->aapWait)
		sendReply(info, MType::DDelta);
	// broadcast new parameter in main thread
}

void Master::handleDeltaPap(const std::string& data, const RPCInfo& info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	updateOnlineLoss(get<2>(deltaMsg), s);
	++nDelta;
	rph.input(typeDDeltaAll, s);
	mtDeltaSum += tmr.elapseSd();
}
