#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "math/RandomGenerator.h"
#include "util/Timer.h"
#include <atomic>
#include <random>
#include <mutex>

class Worker : public Runner{
public:
	Worker();
	virtual void init(const ConfData* conf, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	using init_ft = void(Worker::*)();
	using process_ft = void(Worker::*)();
	using handler_ft = void(Worker::*)(const std::string&, const RPCInfo&);
	using lbs_ft = size_t(Worker::*)(const size_t gbs);
	init_ft initFun;
	process_ft processFun;
	handler_ft paramFun;
	lbs_ft lbsFun;
	
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

	void papOnlineProbe1();
	void papOnlineProbe2();
	void papOnlineProbe3();
	void papOnlineProbe4();
	void papOnlineProbeBenchmark();
	void papOnlineProbeFile();

// local logic
private:
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
	void sendDelta(std::vector<double>& delta, const size_t cnt, const double loss);

	void initializeParameter();
	void bufferParameter(Parameter& p);
	void applyBufferParameter(); // using the buffer
	void waitParameter();
	void fetchParmeter();

	// calculate loss with data in range [start, start+cnt] using current model parameter
	double calcLoss(const size_t start, const size_t cnt);
	void sendLoss(const double loss);

	void sendReport(const std::vector<double>& report); // pap progress

	void pauseTrain();
	void resumeTrain();

	size_t calcLocalBatchSizeDivide(const size_t gbs);
	size_t calcLocalBatchSizeWhole(const size_t gbs);

	void initSpeedAdjustment();
	double getSpeedFactor();

// singal
public:
	void handleNormalControl(const std::string& data, const RPCInfo& info);
	void handleImmediateControl(const std::string& data, const RPCInfo& info);

	void handleReply(const std::string& data, const RPCInfo& info);
	void handleWorkerList(const std::string& data, const RPCInfo& info);
	void handleStart(const std::string& data, const RPCInfo& info);
	void handlePause(const std::string& data, const RPCInfo& info);
	void handleContinue(const std::string& data, const RPCInfo& info);
	void handleDeltaRequest(const std::string& data, const RPCInfo& info);
	void handleLossRequest(const std::string& data, const RPCInfo& info);
	void handleReset(const std::string& data, const RPCInfo& info);
	void handleMetaConf(const std::string& data, const RPCInfo& info);

	void handleTerminate(const std::string& data, const RPCInfo& info);
	void handleProbeDone(const std::string& data, const RPCInfo& info);

	void handleParameterProbe(const std::string& data, const RPCInfo& info);
	void handleParameter(const std::string& data, const RPCInfo& info); // bsp and tap
	void handleParameterSsp(const std::string& data, const RPCInfo& info); // ssp and sap
	void handleParameterFsp(const std::string& data, const RPCInfo& info);
	void handleParameterAap(const std::string& data, const RPCInfo& info);
	void handleParameterPap(const std::string& data, const RPCInfo& info);
		
private:
	size_t dataPointer;
	size_t localBatchSize;
	size_t localReportSize; // pap
	int ln; // log-every-n times

	int masterNID; // network id of master
	IDMapper wm; // worker mapper
	SyncUnit suOnline;
	SyncUnit suDatasetInfo;
	SyncUnit suStart;
	SyncUnit suConf;

	Timer tmrTrain;

	SyncUnit suLossReq;
	size_t lossReqStart, lossReqCount; // the data points to be used for calculating loss
	
	bool hasNewParam;
	std::mutex mParam;
	Parameter bfParam;
	SyncUnit suParam;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta;
	//size_t bfDeltaDpCount; // the number of data points used for current bfDelta
	bool requestingDelta; // pap: whether a delat is being requested now

	int iterParam;

	size_t n_report; // pap: moniter report processing time
	double t_report;

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;

	// speed adjustment
	std::vector<std::pair<double, double>> speedAdjList;
	RandomGenerator speedFactor;

	// online probe
	SyncUnit suProbeDone;
};
