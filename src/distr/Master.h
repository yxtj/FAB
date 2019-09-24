#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "IntervalEstimator.h"
#include "ReceiverSelector.h"
#include "model/ParamArchiver.h"
#include "driver/tools/SyncUnit.h"
#include "util/Timer.h"
#include <vector>
#include <fstream>
#include <mutex>
#include <atomic>
#include <map>

class Master : public Runner{
public:
	Master();
	virtual void init(const ConfData* conf, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	using init_ft = void(Master::*)();
	using process_ft = void(Master::*)();
	using handler_ft = void(Master::*)(const std::string&, const RPCInfo&);
	init_ft initFun;
	process_ft processFun;
	handler_ft deltaFun;

	callback_t localCBBinder(handler_ft fp);
	void bindMode();
	void probeModeInit();
	void probeModeProcess();

// parallel modes
private:
	void bspInit();
	void bspProcess();
	void tapInit();
	void tapProcess();
	void sspInit();
	void sspProcess();
	void sapInit();
	void sapProcess();
	void fspInit();
	void fspProcess();
	void aapInit();
	void aapProcess();
	void papInit();
	void papProcess();
	void papProbe();
	void pap2Process();
	void pap2Probe();

// local logic
private:
	void applyDelta(std::vector<double>& delta, const int source); // only apply
	void clearAccumulatedDelta(); // only restore
	void accumulateDelta(const std::vector<double>& delta, const size_t cnt); // only update
	void applyDeltaNext(const int d); // at slot d
	void clearAccumulatedDeltaNext(const int d); // include slot d, also set bfDelta as slot[d+1]
	void shiftAccumulatedDeltaNext(); // optimized version of clearAccumulatedDeltaNext(0)
	void accumulateDeltaNext(const int d, const std::vector<double>& delta, const size_t cnt); // include slot d
	//void receiveDelta(std::vector<double>& delta, const int source);
	
	void setTerminateCondition(const double time = 0.0,
		const size_t nPoint = 0, const size_t nDelta = 0, const size_t nIter = 0); // 0 means unlimited
	bool terminateCheck();
	void checkDataset();

	void initializeParameter(); // generate initialized parameter locally
	void coordinateParameter(); // coordinate initialized parameter
	void sendParameter(const int target);
	void broadcastParameter();
	void multicastParameter(const int source);
	void waitParameterConfirmed();
	void broadcastReset(const int iter, const Parameter& p);

	bool needArchive();
	void archiveProgress(const bool force = false);

	size_t estimateGlobalBatchSize();
	size_t optFkGlobalBatchSize(); // compute opt k from f(k)
	size_t estimateLocalReportSize(const bool quick = false);

	void updateOnlineLoss(const int source, const double loss);
	void updateIterationTime(const int src, const double time);
	void commonHandleDelta(const int src, const size_t n, const double loss, const double time);

// signal logic
public:
	void broadcastWorkerList();
	void waitReady();
	void broadcastStart();
	void broadcastSignalPause();
	void broadcastSignalContinue();
	void broadcastSignalTerminate();
	// global batch size, local report size. send 0 means keeping the old one
	void broadcastSizeConf(const size_t gbs, const size_t lrs);
	void waitDeltaFromAny(); // dont reset suDeltaAny
	void waitDeltaFromAll(); // reset suDeltaAll
	void gatherDelta();
	double gatherLoss();

// handler
public:
	void handleNormalControl(const std::string& data, const RPCInfo& info);
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleOnline(const std::string& data, const RPCInfo& info);
	void handleDataset(const std::string& data, const RPCInfo& info);
	void handleReady(const std::string& data, const RPCInfo& info);

	void handleImmediateControl(const std::string& data, const RPCInfo& info);
	void handleClosed(const std::string& data, const RPCInfo& info);

	void handleParameter(const std::string& data, const RPCInfo& info);
	void handleReport(const std::string& data, const RPCInfo& info);
	void handleLoss(const std::string& data, const RPCInfo& info);

	void handleDeltaProbe(const std::string& data, const RPCInfo& info);

	void handleDeltaBsp(const std::string& data, const RPCInfo& info);
	void handleDeltaTap(const std::string& data, const RPCInfo& info);
	void handleDeltaSsp(const std::string& data, const RPCInfo& info);
	void handleDeltaSap(const std::string& data, const RPCInfo& info);
	void handleDeltaFsp(const std::string& data, const RPCInfo& info);
	void handleDeltaAap(const std::string& data, const RPCInfo& info);
	void handleDeltaPap(const std::string& data, const RPCInfo& info);
	void handleDeltaPap2(const std::string& data, const RPCInfo& info);
	void handleDeltaTail(const std::string& data, const RPCInfo& info);
	void handleDeltaIgnore(const std::string& data, const RPCInfo& info);

private:
	std::vector<double>  bfDelta;
	size_t bfDeltaDpCount; // the number of data points used for current bfDelta
	std::vector<int> deltaIter; // the number of delta received from each source
	std::vector<std::vector<double>> bfDeltaNext; // buffer the delta for further (offset 0 is bfDelta, so left empty)
	std::vector<size_t> bfDeltaDpCountNext;
	std::mutex mbfd; // mutex for bdDelta, bfDeltaNext

	IDMapper wm; // worker id mapper
	double factorDelta;
	size_t nx, ny; // length of x and y
	std::vector<size_t> nPointWorker; // number of data-points on each worker
	size_t nPointTotal;

	IntervalEstimator* pie; // for flexible parallel modes
	ReceiverSelector* prs;
	std::atomic<int> lastDeltaSource;

	int ln; // log-every-n times

	double termTime;
	size_t termIter, termDelta, termPoint;

	//size_t iter; // [defined in Runner] current iteration being executate now (not complete)
	size_t nPoint; // # of used data point by now
	double mtDeltaSum; // master side time for processing all delta
	size_t nDelta; // # of received delta, usually used for asynchronous cases
	double mtParameterSum; // master side time for sending all parameters
	double mtOther; // time other than processing/sending parameter, delta and report. include: archive, log
	double lossOnline, lossGlobal; // the loss for one global batch
	std::vector<double> lastDeltaLoss;

	double timeOffset; // used for accounting time if resumed from previous archive
	Timer tmrTrain;

	// progressive async
	double mtReportSum; // master side time for processing all reports
	size_t nReport; // master side # of processed reports
	std::mutex mReportProc;
	std::vector<int> reportProcEach; // how many data point is processed
	int reportProcTotal;
	SyncUnit suPap; // reported count reached a batch
	size_t localReportSize;
	size_t globalBatchSize;

	Parameter initP; // cache init parameter for probe
	std::map<size_t, double> gkProb; // cache probed gk
	double wtu;

	std::vector<double> wtIteration; // worker side time per iteration (interval between delta reports)
	std::vector<double> wtIterLast; // time of receiving the latest delta report

	std::vector<double> wtDatapoint; // worker side time per data point
	std::vector<double> wtDelta; // worker side time per delta sending
	std::vector<double> wtReport;  // worker side time per report sending

	SyncUnit suOnline;
	SyncUnit suWorker;
	SyncUnit suAllClosed;
	SyncUnit suDatasetInfo;
	SyncUnit suReady;
	SyncUnit suHyper; // hyper parameters (global batch size, local report size) 
	SyncUnit suLoss;
	int typeDDeltaAny, typeDDeltaAll;
	SyncUnit suDeltaAny, suDeltaAll;
	SyncUnit suParam; // reply of parameter broadcast
	SyncUnit suTPause, suTContinue;

	// archive
	bool doArchive;
	ParamArchiver archiver;
	size_t lastArchIter;
	Timer tmrArch;
	bool archDoing;

	// probe
	size_t probeNeededDelta, probeNeededPoint, probeNeededIter;
	bool probeReached;
};
