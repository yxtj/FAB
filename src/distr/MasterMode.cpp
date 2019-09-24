#include "Master.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "math/accumulate.h"
#include <algorithm>
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
		broadcastReset(0, p0);
		double time = tmr.elapseSd();
		gatherLoss();
		double rate = lossOnline / time;
		LOG(INFO) << "batch size: " << gbs << " gain: " << lossOnline << " rate: " << rate;
	}
}

void Master::handleDeltaProbe(const std::string& data, const RPCInfo& info)
{
	(this->*deltaFun)(data, info);
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
	regDSPProcess(MType::DReport, localCBBinder(&Master::handleReportPap));
}

void Master::papProcess()
{
	papProbe();
	VLOG(1) << "Finish prob with gbs= " << globalBatchSize;
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
			VLOG(2) << "gbs=" << globalBatchSize << "\tlrs=" << localReportSize;
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

void Master::papProbe()
{
	suLoss.wait_n_reset();
	double loss0 = lossOnline;
	vector<size_t> gbsList = { globalBatchSize, globalBatchSize / 2, globalBatchSize / 4 };
	vector<double> lossList(gbsList.size(), loss0);
	vector<double> gainList(gbsList.size(), 0.0);
	int loops = 10;
	while(--loops > 0){
		for(size_t i = 0; i < gbsList.size(); ++i){
			size_t gbs = gbsList[i];
			broadcastSizeConf(gbs, 0);
			suDeltaAll.wait_n_reset();
			double loss = lossOnline;
			gainList[i] += lossList[i] - loss;
			lossList[i] = loss;
		}
	}
	auto it = max_element(gainList.begin(), gainList.end());
	globalBatchSize = gbsList[it-gainList.begin()];
}

// ---- pap2

void Master::pap2Process()
{
	VLOG(1) << "Start prob phase with gbs=" << globalBatchSize;
	if(conf->papDynamicBatchSize) {
		pap2Probe();
		size_t gbs = max(optFkGlobalBatchSize(), estimateGlobalBatchSize());
		if(gbs != globalBatchSize){
			globalBatchSize = gbs;
			localReportSize = globalBatchSize / nWorker / 2;
			broadcastSizeConf(globalBatchSize, localReportSize);
		}
	}
	VLOG(1) << "Finish prob phase with gbs=" << globalBatchSize << " time=" << tmrTrain.elapseSd() << " gkProb:" << gkProb;
	
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		// VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
		VLOG(2) << "Start iteration: " << iter << " upd: " << nPoint;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			// VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl);
			tl = t;
			double mtu = mtDeltaSum / nDelta;
			double mtb = mtParameterSum / stat.n_par_send;
			double mtr = mtReportSum / nReport;
			double mto = mtOther / iter;

			double wtd = hmean(wtDatapoint);
			double wtc = mean(wtDelta);
			double wtr = mean(wtReport);

			VLOG(2) << "mtu=" << mtu << "\tmtb=" << mtb << "\tmtr=" << mtr << "\tmto=" << mto
				<< "\twtd=" << wtd << "\twtc=" << wtc << "\twtr=" << wtr << "\tloss=" << lossOnline;
		}
		mtOther += tmr.elapseSd();
		// wait until the report counts reach a global mini batch
		suPap.wait_n_reset();
		//// online change globalBatchSize
		if(conf->papDynamicBatchSize) {
			size_t ogbs = optFkGlobalBatchSize();
			size_t egbs = estimateGlobalBatchSize();
			size_t old_gbs = globalBatchSize;
			globalBatchSize = max(ogbs, egbs);
			size_t olrs = globalBatchSize / nWorker / 2;
			size_t elrs = estimateLocalReportSize();
			size_t old_lrs = localReportSize;
			if(conf->papDynamicReportFreq){
				localReportSize = max(olrs, elrs);
			}
			VLOG_IF(old_gbs != globalBatchSize || old_lrs != localReportSize, 2)
				<< "gbs=" << globalBatchSize << " (o=" << ogbs << ", e=" << egbs << ")"
				<< " lrs=" << localReportSize << "(o=" << olrs << ", e=" << elrs << ")";
			broadcastSizeConf(globalBatchSize, localReportSize);
		}
		// VLOG(2) << "gather Delta";
		gatherDelta();
		stat.t_dlt_wait += tmr.elapseSd();
		// VLOG(2) << "received Delta " << nDelta << " with " << nPoint;
		broadcastParameter();

		tmr.restart();
		archiveProgress();
		++iter;
		mtOther += tmr.elapseSd();
	}
}

/// pap with online probe
void Master::pap2Probe()
{
	const double toleranceFactor = 0.8;
	double tl = tmrTrain.elapseSd();
	double maxfk = -1;
	probeReached = false;
	size_t probeSize = static_cast<size_t>(nPointTotal * conf->probeRatio);
	size_t lastNPoint = nPoint;

	suLoss.wait_n_reset();
	double lastLoss = lossOnline;
	VLOG(2) << "loss_0 = " << lossOnline;

	double lastProbeTime = tmrTrain.elapseSd();
	while(!terminateCheck() && !probeReached){
		Timer tmr;
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

			VLOG(2) << "mtu=" << mtu << "\tmtb=" << mtb << "\tmtr=" << mtr << "\tmto=" << mto
				<< "\twtd=" << wtd << "\twtc=" << wtc << "\twtr=" << wtr << "\tloss=" << lossOnline;
		}
		mtOther += tmr.elapseSd();
		// wait until the report counts reach a global mini batch
		suPap.wait_n_reset();

		gatherDelta();
		stat.t_dlt_wait += tmr.elapseSd();

		/// reset gbs
		if (conf->papDynamicBatchSize && (nPoint- lastNPoint) > probeSize) {
			double gk = (lastLoss - lossOnline) / globalBatchSize;
			lastLoss = lossOnline;
			gkProb[globalBatchSize] = gk;
			double wtd = hmean(wtDatapoint);
			double wtc = mean(wtDelta);
			double tk1 =(wtd/nWorker + wtc/globalBatchSize);
			double tk2 = tmr.elapseSd() - lastProbeTime;
			double fk = gk / tk2;
			double mink = estimateGlobalBatchSize();
			double lastTTloss = sum(lastDeltaLoss);

			VLOG(2) << "PROB k=" << globalBatchSize << "\ttk=" << tk1 << ", " << tk2
				<< "\tnp=" << nPoint << "\tgk=" << gk << "\tfk=" << gk / tk1 << ", " << gk / tk2 << "\n"
				<< "wtd=" << wtd << ", " << wtDatapoint << "\twtu=" << wtc << ", " << wtDelta << "\n"
				<< "maxfk=" << maxfk << "\tmink=" << mink << "\tonlineloss=" << lossOnline << "\tsumloss=" << lossGlobal
				<< "\tunitloss=" << lossGlobal/nPoint << "\tlastTTloss=" << lastTTloss
				<< "\tlastunitLoss=" << lastTTloss/globalBatchSize;
		
			if (maxfk < 0 || fk > maxfk * toleranceFactor) {
				maxfk = max(fk, maxfk);
				if (globalBatchSize / 2 < mink) {
					break;
				}
				globalBatchSize /= 2;
				// model.setParameter(initP);
				localReportSize = globalBatchSize/nWorker/2;
				broadcastSizeConf(globalBatchSize, localReportSize);
			} else {
				// TODO:
				//globalBatchSize *= 2; 
				//localReportSize = globalBatchSize/nWorker/2;
				//broadcastSizeConf(globalBatchSize, localReportSize);
				// probeReached = true;
				break;
			}
			// iter = 0;
			lastNPoint = nPoint;
			lossGlobal = 0;
		}

		broadcastParameter();

		tmr.restart();
		archiveProgress();
		++iter;
		mtOther += tmr.elapseSd();
	}
}

void Master::handleReportPap(const std::string& data, const RPCInfo& info)
{
	const double emaFactor = 0.8;
	Timer tmr;
	vector<double> report = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int wid = wm.nid2lid(info.source);
	bool flag = false;
	// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
	{
		lock_guard<mutex> lg(mReportProc);
		reportProcEach[wid] += static_cast<int>(report[0]);
		reportProcTotal += static_cast<int>(report[0]);

		//if(conf->papSearchBatchSize || conf->papSearchReportFreq){
		wtDatapoint[wid] = (1 - emaFactor) * wtDatapoint[wid] + emaFactor * report[1];
		wtDelta[wid] = (1 - emaFactor) * wtDelta[wid] + emaFactor * report[2];
		wtReport[wid] = (1 - emaFactor) * wtDelta[wid] + emaFactor * report[3];
		//}
		// updateOnlineLoss(wid, report[4]); // estimated loss
		lossGlobal += report[4]; /// accumulate loss
		if(report.size() > 5)
			wtu = wtu == 0 ? report[5] : (wtu * (nWorker - 1) + report[5]) / nWorker; /// worker update time
		if(reportProcTotal >= globalBatchSize) {
			suPap.notify();
			reportProcTotal = 0; // request delta and reset counter
		}
		// VLOG(2) << "Recieve report from: " << wid << " with " << report
		// 	<< "\ttt: " << reportProcTotal << "\tgbs: " << globalBatchSize;
	}
	++nReport;
	mtReportSum += tmr.elapseSd();
}

// ---- handlers ----

void Master::handleDeltaBsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	applyDelta(get<1>(deltaMsg), s);

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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	applyDelta(get<1>(deltaMsg), s);

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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	applyDelta(get<1>(deltaMsg), s);

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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	//applyDelta(get<1>(deltaMsg), s);
	accumulateDelta(get<1>(deltaMsg), n); // applied in the main process

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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	applyDelta(get<1>(deltaMsg), s);

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
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	applyDelta(get<1>(deltaMsg), s);

	rph.input(typeDDeltaAll, s);
	mtDeltaSum += tmr.elapseSd();
}

void Master::handleDeltaPap2(const std::string& data, const RPCInfo& info)
{
	Timer tmr;
	auto deltaMsg = deserialize<tuple<size_t, vector<double>, double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	commonHandleDelta(s, get<0>(deltaMsg), get<2>(deltaMsg), tmrTrain.elapseSd());
	// nPoint += get<0>(deltaMsg);
	applyDelta(get<1>(deltaMsg), s);
	// ++nDelta;

	rph.input(typeDDeltaAll, s);
	mtDeltaSum += tmr.elapseSd();
}
