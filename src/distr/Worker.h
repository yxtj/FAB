#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include <atomic>
#include <mutex>

class Worker : public Runner{
public:
	Worker();
	virtual void init(const ConfData* conf, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Worker::*fp)(const std::string&, const RPCInfo&));
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
	//void generalProcess();

	void updatePointer(const size_t scan, const size_t report);
	void sendOnline();
	void waitWorkerList();
	void sendDatasetInfo();
	void sendReady();
	void waitStart();
	void sendClosed();

	void clearDelta();
	void averageDelta(const size_t size);
	void accumulateDelta(const std::vector<double>& delta);
	void sendDelta(std::vector<double>& delta, const size_t cnt);

	void initializeParameter();
	void bufferParameter(Parameter& p);
	void applyBufferParameter(); // using the buffer
	void waitParameter();
	void fetchParmeter();

	void sendReport(const int cnt); // pap progress

	void pauseTrain();
	void resumeTrain();

// singal
public:
	void handleNormalControl(const std::string& data, const RPCInfo& info);
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleWorkerList(const std::string& data, const RPCInfo& info);
	void handleStart(const std::string& data, const RPCInfo& info);
	void handlePause(const std::string& data, const RPCInfo& info);
	void handleContinue(const std::string& data, const RPCInfo& info);
	void handleDeltaRequest(const std::string& data, const RPCInfo& info);

	void handleImmediateControl(const std::string& data, const RPCInfo& info);
	void handleTerminate(const std::string& data, const RPCInfo& info);

	void handleParameter(const std::string& data, const RPCInfo& info);
	void handleParameterSsp(const std::string& data, const RPCInfo& info);
	void handleParameterFsp(const std::string& data, const RPCInfo& info);
	void handleParameterAap(const std::string& data, const RPCInfo& info);
	void handleParameterPap(const std::string& data, const RPCInfo& info);
		

private:
	size_t dataPointer;
	size_t localBatchSize;
	int ln; // log-every-n times

	int masterNID; // network id of master
	IDMapper wm; // worker mapper
	SyncUnit suOnline;
	SyncUnit suDatasetInfo;
	SyncUnit suStart;
	
	bool hasNewParam;
	std::mutex mParam;
	Parameter bfParam;
	SyncUnit suParam;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta;
	//size_t bfDeltaDpCount; // the number of data points used for current bfDelta
	bool requestingDelta; // pap: whether a delat is being requested now

	int iterParam;

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;
};
