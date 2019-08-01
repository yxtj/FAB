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

class Master : public Runner{
public:
	Master();
	virtual void init(const ConfData* conf, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Master::*fp)(const std::string&, const RPCInfo&));
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
	
	bool terminateCheck();
	void checkDataset();

	void initializeParameter(); // generate initialized parameter locally
	void coordinateParameter(); // coordinate initialized parameter
	void sendParameter(const int target);
	void broadcastParameter();
	void multicastParameter(const int source);
	void waitParameterConfirmed();

	bool needArchive();
	void archiveProgress(const bool force = false);

// signal logic
public:
	void broadcastWorkerList();
	void waitReady();
	void broadcastStart();
	void broadcastSignalPause();
	void broadcastSignalContinue();
	void broadcastSignalTerminate();
	void waitDeltaFromAny(); // dont reset suDeltaAny
	void waitDeltaFromAll(); // reset suDeltaAll
	void gatherDelta();

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

	void handleDelta(const std::string& data, const RPCInfo& info);
	void handleDeltaTap(const std::string& data, const RPCInfo& info);
	void handleDeltaSsp(const std::string& data, const RPCInfo& info);
	void handleDeltaSap(const std::string& data, const RPCInfo& info);
	void handleDeltaFsp(const std::string& data, const RPCInfo& info);
	void handleDeltaAap(const std::string& data, const RPCInfo& info);
	void handleDeltaTail(const std::string& data, const RPCInfo& info);

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
	size_t nPoint;

	IntervalEstimator* pie; // for flexible parallel modes
	ReceiverSelector* prs;
	std::atomic<int> lastDeltaSource;

	int ln; // log-every-n times

	//size_t iter; // [defined in Runner] current iteration being executate now (not complete)
	size_t nUpdate; // used for Async case
	double timeOffset; // used for accounting time if resumed
	Timer tmrTrain;

	SyncUnit suOnline;
	SyncUnit suWorker;
	SyncUnit suAllClosed;
	SyncUnit suDatasetInfo;
	SyncUnit suReady;
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
};
