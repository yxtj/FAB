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
	wtDatapoint.assign(nWorker, 9e99);
	wtDelta.assign(nWorker, 0.0);
	wtReport.assign(nWorker, 0.0);
	//}
	suPap.reset();
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaPap));
	regDSPProcess(MType::DReport, localCBBinder(&Master::handleReportPap));
}

// ---- pap2

void Master::papProcess()
{
	if(conf->papOnlineProbeVersion > 0){
		VLOG(1) << "Start probe phase with gbs=" << globalBatchSize;
		if(conf->papOnlineProbeVersion == 1)
			papOnlineProbe1();
		else if(conf->papOnlineProbeVersion == 2)
			papOnlineProbe2();
		else if(conf->papOnlineProbeVersion == 3)
			papOnlineProbe3();
		else if(conf->papOnlineProbeVersion == 4)
			papOnlineProbe4();
		else
			LOG(FATAL) << "Online probe version " << conf->papOnlineProbeVersion << " not supported";

		size_t gbs = max(optFkGlobalBatchSize(), estimateMinGlobalBatchSize());
		if(gbs != globalBatchSize){
			globalBatchSize = gbs;
			localReportSize = globalBatchSize / nWorker / 2;
			broadcastSizeConf(globalBatchSize, localReportSize);
		}
		VLOG(1) << "Finish probe phase with gbs=" << globalBatchSize << " time=" << tmrTrain.elapseSd() << " gkProb:" << gkProb;
	}
	
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter << " data-points: " << nPoint;
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
			size_t gbs = estimateMinGlobalBatchSize();

			VLOG(2) << "min-gbs=" << gbs << "\tloss-est=" << lossOnline << "\tloss-last="<<sum(lastDeltaLoss)
				<< "\tmtu=" << mtu << "\tmtb=" << mtb << "\tmtr=" << mtr << "\tmto=" << mto
				<< "\twtd=" << wtd << "\twtc=" << wtc << "\twtr=" << wtr;
		}
		mtOther += tmr.elapseSd();
		// wait until the report counts reach a global mini batch
		suPap.wait_n_reset();
		//// online change globalBatchSize
		if(conf->papDynamicBatchSize) {
			size_t ogbs = optFkGlobalBatchSize();
			size_t egbs = estimateMinGlobalBatchSize();
			size_t old_gbs = globalBatchSize;
			globalBatchSize = max(ogbs, egbs);
			size_t olrs = globalBatchSize / nWorker / 2;
			size_t elrs = estimateMinLocalReportSize();
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

void Master::papOnlineProbe1()
{
	const double toleranceFactor = 0.8;
	double tl = tmrTrain.elapseSd();
	double maxfk = -1;
	probeReached = false;
	size_t probeSize = static_cast<size_t>(nPointTotal * conf->probeRatio);

	suLoss.wait_n_reset();
	double lastLoss = lossOnline / globalBatchSize;
	VLOG(2) << "loss_0 = " << lastLoss << "\tprobeSize=" << probeSize;

	size_t lastProbeNPoint = nPoint;
	int lastProbeIter = iter;
	double lastProbeTime = tmrTrain.elapseSd();
	while(!terminateCheck() && !probeReached){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl) << " data-points: " << nPoint - lastProbeNPoint;
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
		broadcastParameter();
		tmr.restart();
		archiveProgress();
		++iter;
		mtOther += tmr.elapseSd();

		/// reset gbs
		if ((nPoint- lastProbeNPoint) > probeSize) {
			double nloss1 = lossOnline / globalBatchSize;
			double nloss2 = sum(lastDeltaLoss) / (nPoint - lastProbeNPoint);
			double nloss3 = lossGlobal / globalBatchSize;
			double nloss = max(max(nloss1, nloss2), nloss3);
			double gk = lastLoss - nloss;
			//lastLoss = nloss1 / globalBatchSize;
			lastLoss = nloss2 / globalBatchSize;
			gkProb[globalBatchSize] = gk;
			double wtd = hmean(wtDatapoint);
			double wtc = mean(wtDelta);
			double tk1 =(wtd/nWorker + wtc/globalBatchSize);
			double tk2 = tmrTrain.elapseSd() - lastProbeTime;
			double fk = gk / tk2;
			size_t mink = estimateMinGlobalBatchSize();

			VLOG(2) << "probe k=" << globalBatchSize << "\titer=" << iter - lastProbeIter << "\tnp=" << nPoint - lastProbeNPoint
				<< "\ttk=" << tk1 << ", " << tk2 << "\tgk=" << gk << "\tfk=" << gk / tk1 << ", " << gk / tk2 << "\n"
				<< "wtd=" << wtd << ", " << wtDatapoint << "\twtc=" << wtc << ", " << wtDelta << "\n"
				<< "maxfk=" << maxfk << "\tmink=" << mink << "\tunit-loss-online=" << nloss1
				<< "\tunit-loss-sum=" << nloss3 << "\tunit-loss-recent=" << nloss2;
		
			if (maxfk < 0 || fk > maxfk * toleranceFactor) {
				maxfk = max(fk, maxfk);
				if(globalBatchSize / 2 >= mink) {
					globalBatchSize /= 2;
					localReportSize = globalBatchSize / nWorker / 2;
					broadcastSizeConf(globalBatchSize, localReportSize);
					lossOnline /= 2;
					for(auto& v : lastDeltaLoss)
						v /= 2;
				} else{
					probeReached = true;
				}
			} else {
				// TODO:
				//globalBatchSize *= 2; 
				//localReportSize = globalBatchSize/nWorker/2;
				//broadcastSizeConf(globalBatchSize, localReportSize);
				probeReached = true;
				//break;
			}
			lastProbeNPoint = nPoint;
			lastProbeIter = iter;
			lastProbeTime = tmrTrain.elapseSd();
			lossGlobal = 0;
		}
	}
}

void Master::papOnlineProbe2()
{
	// normal run
	// if probe size reached
		// first request loss0
		// estimate gbs
		// check gbs condition
			// if so send conf
			// else send probe done

	const double toleranceFactor = 0.8;
	double tl = tmrTrain.elapseSd();
	double maxfk = -1;
	probeReached = false;
	size_t probeSize = static_cast<size_t>(nPointTotal * conf->probeRatio);

	suLoss.wait_n_reset();
	double lastLoss = lossOnline / globalBatchSize;
	VLOG(2) << "loss_0 = " << lastLoss << "\tprobeSize=" << probeSize;

	size_t lastProbeNPoint = nPoint;
	int lastProbeIter = iter;
	double lastProbeTime = tmrTrain.elapseSd();
	while(!terminateCheck() && !probeReached){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % ln == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent " << ln << " iterations: " << (t - tl) << " data-points: " << nPoint - lastProbeNPoint;
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
		broadcastParameter();
		tmr.restart();
		archiveProgress();
		++iter;
		mtOther += tmr.elapseSd();

		/// reset gbs
		if ((nPoint- lastProbeNPoint) > probeSize) {
			double nloss1 = lossOnline / globalBatchSize;
			double nloss2 = sum(lastDeltaLoss) / (nPoint - lastProbeNPoint);
			double nloss3 = lossGlobal / globalBatchSize;
			double nloss = max(max(nloss1, nloss2), nloss3);
			double gk = lastLoss - nloss;
			//lastLoss = nloss1 / globalBatchSize;
			lastLoss = nloss2 / globalBatchSize;
			gkProb[globalBatchSize] = gk;
			double wtd = hmean(wtDatapoint);
			double wtc = mean(wtDelta);
			double tk1 =(wtd/nWorker + wtc/globalBatchSize);
			double tk2 = tmrTrain.elapseSd() - lastProbeTime;
			double fk = gk / tk2;
			size_t mink = estimateMinGlobalBatchSize();

			VLOG(2) << "probe k=" << globalBatchSize << "\titer=" << iter - lastProbeIter << "\tnp=" << nPoint - lastProbeNPoint
				<< "\ttk=" << tk1 << ", " << tk2 << "\tgk=" << gk << "\tfk=" << gk / tk1 << ", " << gk / tk2 << "\n"
				<< "wtd=" << wtd << ", " << wtDatapoint << "\twtc=" << wtc << ", " << wtDelta << "\n"
				<< "maxfk=" << maxfk << "\tmink=" << mink << "\tunit-loss-online=" << nloss1
				<< "\tunit-loss-sum=" << nloss3 << "\tunit-loss-recent=" << nloss2;
		
			if (maxfk < 0 || fk > maxfk * toleranceFactor) {
				maxfk = max(fk, maxfk);
				if(globalBatchSize / 2 >= mink) {
					globalBatchSize /= 2;
					localReportSize = globalBatchSize / nWorker / 2;
					broadcastSizeConf(globalBatchSize, localReportSize);
					lossOnline /= 2;
					for(auto& v : lastDeltaLoss)
						v /= 2;
				} else{
					probeReached = true;
				}
			} else {
				// TODO:
				//globalBatchSize *= 2; 
				//localReportSize = globalBatchSize/nWorker/2;
				//broadcastSizeConf(globalBatchSize, localReportSize);
				probeReached = true;
				//break;
			}
			lastProbeNPoint = nPoint;
			lastProbeIter = iter;
			lastProbeTime = tmrTrain.elapseSd();
			lossGlobal = 0;
		}
	}

}

void Master::papOnlineProbe3()
{}

void Master::papOnlineProbe4()
{}

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
		wtDatapoint[wid] = wtDatapoint[wid] == 9e99 ? report[1] : (1 - emaFactor) * wtDatapoint[wid] + emaFactor * report[1];
		wtDelta[wid] = wtDelta[wid] == 0.0 ? report[2] : (1 - emaFactor) * wtDelta[wid] + emaFactor * report[2];
		wtReport[wid] = wtReport[wid] == 0.0 ? report[3] : (1 - emaFactor) * wtDelta[wid] + emaFactor * report[3];
		// updateOnlineLoss(wid, report[4]); // estimated loss
		lossGlobal += report[4]; /// accumulate loss
		if(report.size() > 5)
			wtu = wtu == 0 ? report[5] : (wtu * (nWorker - 1) + report[5]) / nWorker; /// worker update time
		//}
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
