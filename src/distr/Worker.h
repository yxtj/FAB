#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include <atomic>
#include <mutex>

class Worker : public Runner{
public:
	Worker();
	virtual void init(const Option* opt, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Worker::*fp)(const std::string&, const RPCInfo&));
	void bspInit();
	void bspProcess();
	void tapInit();
	void tapProcess();
	void fspInit();
	void fspProcess();
	void aapInit();
	void aapProcess();
	//void generalProcess();

	void updatePointer(const size_t used);
	void sendOnline();
	void waitWorkerList();
	void sendDatasetInfo();
	void sendClosed();

	void accumulateDelta(const std::vector<double>& delta);
	void sendDelta(std::vector<double>& delta, const size_t cnt);
	void bufferParameter(Parameter& p);
	void applyBufferParameter(); // using the buffer
	void waitParameter();
	void fetchParmeter();

	void pauseTrain();
	void resumeTrain();

// singal
public:
	//void handleDelta(const std::string& data, const RPCInfo& info);
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleWorkerList(const std::string& data, const RPCInfo& info);
	void handleParameter(const std::string& data, const RPCInfo& info);
	void handleParameterFsp(const std::string& data, const RPCInfo& info);
	void handleParameterAap(const std::string& data, const RPCInfo& info);
	void handlePause(const std::string& data, const RPCInfo& info);
	void handleContinue(const std::string& data, const RPCInfo& info);
	void handleTerminate(const std::string& data, const RPCInfo& info);
		

private:
	size_t dataPointer;
	size_t localBatchSize;
	int ln; // log-every-n times

	int masterNID; // network id of master
	IDMapper wm; // worker mapper
	SyncUnit suOnline;
	SyncUnit suDatasetInfo;
	
	bool hasNewParam;
	std::mutex mParam;
	Parameter bfParam;
	SyncUnit suParam;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta;
	size_t bfDeltaDpCount; // the number of data points used for current bfDelta

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;
};
