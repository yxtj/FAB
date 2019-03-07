#include "Master.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
using namespace std;

// ---- bulk synchronous parallel

void Master::bspInit()
{
	factorDelta = 1.0 / nWorker;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::bspProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
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
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
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
	deltaIter.assign(nWorker, 0.0);
}

void Master::sspProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
			newIter = false;
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		stat.t_dlt_wait += tmr.elapseSd();
		int p;
		{
			lock_guard<mutex> lg(mbfd);
			p = *min_element(deltaIter.begin(), deltaIter.end());
		}
		if(iter != p){
			{
				lock_guard<mutex> lg(mbfd);
				applyDelta(bfDelta, -1);
				adoptDeltaNext(p - iter);
			}
			// if multiple iteration is passed, send one parameter for each
			for(int i = iter; i < p; ++i)
				broadcastParameter();
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

// ---- staleness asynchronous parallel

void Master::sapInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaSap));
	deltaIter.assign(nWorker, 0);
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
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		stat.t_dlt_wait += tmr.elapseSd();
		int p = static_cast<int>(nUpdate / nWorker + 1);
		if(iter != p){
			{
				lock_guard<mutex> lg(mbfd);
				applyDelta(bfDelta, -1);
				clearAccumulatedDelta();
			}
			// if multiple iteration is passed, send one parameter for each
			for(int i = iter; i < p; ++i)
				broadcastParameter();
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
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaFsp));
}

void Master::fspProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		Timer tmr;
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
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
		tmr.restart();
		applyDelta(bfDelta, -1);
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		stat.t_dlt_calc += tmr.elapseSd();
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
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		stat.t_dlt_wait += tmr.elapseSd();
		multicastParameter(lastDeltaSource);
		int p = static_cast<int>(nUpdate / nWorker + 1);
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
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
	applyDelta(deltaMsg.second, s);
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaTap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	applyDelta(deltaMsg.second, s);
	++nUpdate;
	rph.input(typeDDeltaAll, s);
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
		lock_guard<mutex> lg(mbfd);
		accumulateDelta(deltaMsg.second, deltaMsg.first);
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
	{
		deltaIter[s]++;
		int d = deltaIter[s] - iter;
		lock_guard<mutex> lg(mbfd);
		if(d == 0){
			accumulateDelta(deltaMsg.second, deltaMsg.first);
		} else{
			accumulateDeltaNext(d, deltaMsg.second, deltaMsg.first);
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
	applyDelta(deltaMsg.second, s);
	++nUpdate;
	//static vector<int> cnt(nWorker, 0);
	//++cnt[s];
	//VLOG_EVERY_N(ln/10, 1) << "Update: " << nUpdate << " rsp: " << cnt << " r-pkg: " << net->stat_recv_pkg;
	//rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	lastDeltaSource = s;
	if(opt->aapWait)
		sendReply(info);
	++stat.n_dlt_recv;
	// broadcast new parameter in main thread
}
