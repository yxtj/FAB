#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "IntervalEstimator.h"
#include "ReceiverSelector.h"
#include "driver/tools/SyncUnit.h"
#include "util/Timer.h"
#include <vector>
#include <fstream>
#include <mutex>

class Master : public Runner{
public:
	Master();
	virtual void init(const Option* opt, const size_t lid);
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
	void applyDelta(std::vector<double>& delta, const int source);
	void clearAccumulatedDelta();
	void accumulateDelta(const std::vector<double>& delta, const size_t cnt);
	void accumulateDeltaNext(const int d, const std::vector<double>& delta, const size_t cnt);
	void adoptDeltaNext(const int d);
	//void receiveDelta(std::vector<double>& delta, const int source);
	bool terminateCheck();
	void initializeParameter();
	void sendParameter(const int target);
	void broadcastParameter();
	void multicastParameter(const int source);
	void waitParameterConfirmed();
	bool needArchive();
	void archiveProgress(const bool force = false);

// signal logic
public:
	void broadcastWorkerList();
	void broadcastSignalPause();
	void broadcastSignalContinue();
	void broadcastSignalTerminate();
	void waitDeltaFromAny(); // dont reset suDeltaAny
	void waitDeltaFromAll(); // reset suDeltaAll
	void gatherDelta();

// handler
public:
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleOnline(const std::string& data, const RPCInfo& info);
	void handleDataset(const std::string& data, const RPCInfo& info);
	void handleDelta(const std::string& data, const RPCInfo& info);
	void handleDeltaTap(const std::string& data, const RPCInfo& info);
	void handleDeltaSsp(const std::string& data, const RPCInfo& info);
	void handleDeltaSap(const std::string& data, const RPCInfo& info);
	void handleDeltaFsp(const std::string& data, const RPCInfo& info);
	void handleDeltaAap(const std::string& data, const RPCInfo& info);
	void handleDeltaTail(const std::string& data, const RPCInfo& info);

private:
	Parameter param;
	std::vector<double>  bfDelta;
	size_t bfDeltaDpCount; // the number of data points used for current bfDelta
	std::vector<int> deltaIter; // the number of delta received from a source
	std::vector<std::vector<double>> bfDeltaNext; // buffer the delta for further 
	std::vector<size_t> bfDeltaDpCountNext; // buffer the delta for further 
	std::mutex mbfd; // mutex for bdDelta

	IDMapper wm; // worker id mapper
	double factorDelta;
	size_t nx, ny; // length of x and y
	std::vector<size_t> nPointWorker; // number of data-points on each worker
	size_t nPoint;

	IntervalEstimator* pie; // for flexible parallel modes
	ReceiverSelector* prs;
	int lastDeltaSource;

	int ln; // log-every-n times

	//size_t iter; // [defined in Runner] current iteration being executate now (not complete)
	size_t nUpdate; // used for Async case
	Timer tmrTrain;

	size_t lastArchIter;
	Timer tmrArch;

	SyncUnit suOnline;
	SyncUnit suWorker;
	SyncUnit suAllClosed;
	SyncUnit suDatasetInfo;
	int typeDDeltaAny, typeDDeltaAll;
	SyncUnit suDeltaAny, suDeltaAll;
	SyncUnit suParam; // reply of parameter broadcast
	SyncUnit suTPause, suTContinue;

	std::ofstream foutput;
};
