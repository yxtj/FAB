#include "Worker.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"

using namespace std;

// initialize constant / stable data
Worker::Worker() : Runner() {
	masterNID = 0;
	dataPointer = 0;
	iter = 0;
	iterParam = 0;
	localBatchSize = 1;
	//bfDeltaDpCount = 0;
	n_updParam = 0;
	t_updParam = 0;

	hasNewParam = false;
	allowTrain = true;
	exitTrain = false;
	exitRun = false;
}

void Worker::init(const ConfData* conf, const size_t lid)
{
	this->conf = conf;
	nWorker = conf->nw;
	localReportSize = conf->reportSize;
	localID = lid;
	ln = conf->logIter;
	logName = "W"+to_string(localID);
	setLogThreadName(logName);

	bindMode();
	trainer = TrainerFactory::generate(conf->optimizer, conf->optimizerParam);
	LOG_IF(trainer == nullptr, FATAL) << "Trainer is not set correctly";
	model.init(conf->algorighm, conf->algParam);
	initSpeedAdjustment();

	if(!conf->probe){
		(this->*initFun)();
	} else{
		probeModeInit();
	}
}

void Worker::bindDataset(const DataHolder* pdh)
{
	VLOG(1) << "Bind dataset with " << pdh->size() << " data points";
	this->pdh = pdh;
	trainer->bindDataset(pdh);
}

void Worker::run()
{
	LOG(INFO) << "register handlers";
	registerHandlers();
	startMsgLoop(logName+"-MSG");
	LOG(INFO) << "start";
	DLOG(INFO) << "send online message";
	sendOnline();
	DLOG(INFO) << "waiting worker list";
	waitWorkerList();
	DLOG(INFO) << "send dataset info";
	sendDatasetInfo();
	DLOG(INFO) << "initialize parameter";
	trainer->bindModel(&model);
	trainer->prepare();
	initializeParameter();
	DLOG(INFO) << "got init parameter";
	applyBufferParameter();
	DLOG(INFO) << "ready trainer";
	trainer->ready();
	sendReady();
	waitStart();

	localBatchSize = (this->*lbsFun)(conf->batchSize);
	DLOG(INFO) << "start training with mode: " << conf->mode << ", local batch size: " << localBatchSize;
	iter = 1;
	iterParam = 1;
	if(!conf->probe){
		(this->*processFun)();
	} else{
		probeModeProcess();
	}

	DLOG(INFO) << "finish training";
	sendClosed();
	finishStat();
	delete trainer;
	trainer = nullptr;
	showStat();
	stopMsgLoop();
}

Worker::callback_t Worker::localCBBinder(handler_ft fp)
	//void (Worker::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}

void Worker::bindMode()
{
	if(conf->mode == "bsp"){
		initFun = &Worker::bspInit;
		processFun = &Worker::bspProcess;
		paramFun = &Worker::handleParameter;
		lbsFun = &Worker::calcLocalBatchSizeDivide;
	} else if(conf->mode == "tap"){
		initFun = &Worker::tapInit;
		processFun = &Worker::tapProcess;
		paramFun = &Worker::handleParameter;
		lbsFun = &Worker::calcLocalBatchSizeWhole;
	} else if(conf->mode == "ssp"){
		initFun = &Worker::sspInit;
		processFun = &Worker::sspProcess;
		paramFun = &Worker::handleParameterSsp;
		lbsFun = &Worker::calcLocalBatchSizeDivide;
	} else if(conf->mode == "sap"){
		initFun = &Worker::sapInit;
		processFun = &Worker::sapProcess;
		paramFun = &Worker::handleParameterSsp;
		lbsFun = &Worker::calcLocalBatchSizeWhole;
	} else if(conf->mode == "fsp"){
		initFun = &Worker::fspInit;
		processFun = &Worker::fspProcess;
		paramFun = &Worker::handleParameterFsp;
		lbsFun = &Worker::calcLocalBatchSizeDivide;
	} else if(conf->mode == "aap"){
		initFun = &Worker::aapInit;
		processFun = &Worker::aapProcess;
		paramFun = &Worker::handleParameterAap;
		lbsFun = &Worker::calcLocalBatchSizeWhole;
	} else if(conf->mode == "pap"){
		initFun = &Worker::papInit;
		processFun = &Worker::papProcess;
		paramFun = &Worker::handleParameterPap;
		lbsFun = &Worker::calcLocalBatchSizeDivide;
	} else{
		LOG(FATAL) << "Unsupported mode: " << conf->mode;
	}
}

void Worker::registerHandlers()
{
	regDSPProcess(CType::NormalControl, localCBBinder(&Worker::handleNormalControl));
	//regDSPProcess(MType::CReply, localCBBinder(&Worker::handleReply));
	//regDSPProcess(MType::CWorkers, localCBBinder(&Worker::handleWorkerList));
	//regDSPProcess(MType::CTrainPause, localCBBinder(&Worker::handlePause));
	//regDSPProcess(MType::CTrainContinue, localCBBinder(&Worker::handleContinue));

	regDSPProcess(CType::ImmediateControl, localCBBinder(&Worker::handleImmediateControl));
	//regDSPImmediate(MType::CTerminate, localCBBinder(&Worker::handleTerminate));

	//regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	//regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta));
	if(!conf->probe){
		regDSPProcess(MType::DDelta, localCBBinder(paramFun));
	} else{
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleParameterProbe));
	}

	//addRPHAnySU(MType::DParameter, suParam);
	addRPHAnySU(MType::CDataset, suDatasetInfo);
}

void Worker::updatePointer(const size_t scan, const size_t report)
{
	DVLOG(3) << "update pointer from " << dataPointer << " by " << scan;
	dataPointer += scan;
	if(dataPointer >= trainer->pd->size())
		dataPointer = 0;
	stat.n_point += report;
}

void Worker::sendOnline()
{
	net->send(masterNID, CType::NormalControl,
		make_pair(MType::COnline, static_cast<int>(localID)));
}

void Worker::waitWorkerList()
{
	suOnline.wait();
}

void Worker::sendDatasetInfo()
{
	tuple<size_t, size_t, size_t> t{
		trainer->pd->xlength(), trainer->pd->ylength(), trainer->pd->size() };
	net->send(masterNID, CType::NormalControl, make_pair(MType::CDataset, t));
	suDatasetInfo.wait();
}

void Worker::sendReady()
{
	net->send(masterNID, CType::NormalControl, MType::CReady);
}

void Worker::waitStart()
{
	suStart.wait();
}

void Worker::sendClosed()
{
	net->send(masterNID, CType::ImmediateControl,
		make_pair(MType::CClosed, localID));
}

void Worker::clearDelta()
{
	bfDelta.assign(model.paramWidth(), 0.0);
}

void Worker::averageDelta(const size_t size)
{
	for(auto& v : bfDelta)
		v /= size;
}

void Worker::accumulateDelta(const std::vector<double>& delta)
{
	for (size_t i = 0; i < delta.size(); ++i)
		bfDelta[i] += delta[i];
}

void Worker::sendDelta(std::vector<double>& delta, const size_t cnt, const double loss)
{
	DVLOG(3) << "send delta: " << delta;
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	net->send(masterNID, MType::DDelta, make_tuple(move(cnt), move(delta), move(loss)));
	++stat.n_dlt_send;
}

void Worker::initializeParameter()
{
	if(!conf->resume){
		if(model.getKernel()->needInitParameterByData()){
			// model.param is set in trainer->ready()
			net->send(masterNID, MType::DParameter, model.getParameter().weights);
		}
	}
	waitParameter();
}

void Worker::bufferParameter(Parameter & p)
{
	lock_guard<mutex> lk(mParam);
	bfParam = move(p);
	hasNewParam = true;
}

void Worker::applyBufferParameter()
{
	Timer tmr;
	//DLOG(INFO)<<"has new parameter: "<<hasNewParam;
	if(!hasNewParam)
		return;
	//DLOG(INFO)<<"before lock";
	//lock(mParam, mModel);
	lock_guard<mutex> lk(mParam);
	//DLOG(INFO)<<"after lock";
	DVLOG(3) << "apply parameter: " << bfParam.weights;
	model.setParameter(bfParam);
	//mModel.unlock();
	hasNewParam = false;
	//mParam.unlock();
	n_updParam++;
	t_updParam += tmr.elapseSd();
}

void Worker::waitParameter()
{
	suParam.wait();
}

void Worker::fetchParmeter()
{
	net->send(masterNID, MType::DRParameter, localID);
	suParam.wait_n_reset();
	++stat.n_dlt_recv;
}

double Worker::calcLoss(const size_t start, const size_t cnt)
{
	double loss = 0.0;
	// size_t end = min(cnt, pdh->size());
	for(size_t i = 0; i < cnt; ++i){
		size_t dp = (start+i) % pdh->size();
		double l = model.loss(pdh->get(dp));
		loss += l;
	}
	return loss;
}

void Worker::sendLoss(const double loss)
{
	net->send(masterNID, CType::NormalControl, make_pair(MType::DLoss, loss));
}
void Worker::sendLoss(const vector<double> loss)
{
	net->send(masterNID, CType::NormalControl, make_pair(MType::DLoss, loss));
}

void Worker::sendReport(const std::vector<double>& report)
{
	net->send(masterNID, MType::DReport, report);
}

void Worker::pauseTrain()
{
	allowTrain = false;
}

void Worker::resumeTrain()
{
	allowTrain = true;
}

size_t Worker::calcLocalBatchSizeDivide(const size_t gbs)
{
	size_t lbs = gbs / nWorker;
	if(gbs % nWorker > localID)
		++lbs;
	if(lbs <= 0)
		lbs = 1;
	return lbs;
}

size_t Worker::calcLocalBatchSizeWhole(const size_t gbs)
{
	return gbs;
}

void Worker::initSpeedAdjustment()
{
	const double fixSlowFactor = conf->adjustSpeedHetero ? conf->speedHeterogenerity[localID] : 0.0;
	unsigned seed = conf->seed + 123 + localID;

	const vector<string>& param = conf->adjustSpeedRandom ? conf->speedRandomParam : vector<string>();
	DLOG(DEBUG) << param << "," << fixSlowFactor << "," << seed;
	speedFactor.init(param, fixSlowFactor, seed);
}

// handlers

void Worker::handleNormalControl(const std::string & data, const RPCInfo & info)
{
	int type = deserialize<int>(data);
	//const char* p = data.data() + sizeof(int);
	switch(type){
	case MType::CReply:
		handleReply(data.substr(sizeof(int)), info);
		break;
	case MType::CWorkers:
		handleWorkerList(data.substr(sizeof(int)), info);
		break;
	case MType::CStart:
		handleStart(data.substr(sizeof(int)), info);
		break;
	case MType::CTrainPause:
		handlePause(data.substr(sizeof(int)), info);
		break;
	case MType::CTrainContinue:
		handleContinue(data.substr(sizeof(int)), info);
		break;
	case MType::CReset:
		handleReset(data.substr(sizeof(int)), info);
		break;
	case MType::FSizeConf:
		handleMetaConf(data.substr(sizeof(int)), info);
		break;
	case MType::DRDelta:
		handleDeltaRequest(data.substr(sizeof(int)), info);
		break;
	case MType::DRLoss:
		handleLossRequest(data.substr(sizeof(int)), info);
		break;
	case MType::CProbeDone:
		handleProbeDone(data.substr(sizeof(int)), info);
		break;
		// MType::DParameter is handled directly by message type
	}
}

void Worker::handleReply(const std::string & data, const RPCInfo & info) {
	Timer tmr;
	int type = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	pair<bool, int> s = wm.nidTrans(info.source);
	DVLOG(3) << "get reply from " << (s.first ? "W" : "M") << s.second << " type " << type;
	/*static int ndr = 0;
	if(type == MType::DDelta){
		++ndr;
		VLOG_EVERY_N(ln / 10, 1) << "get delta reply: " << ndr;
	}*/
	rph.input(type, s.second);
}

void Worker::handleWorkerList(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto res = deserialize<vector<pair<int, int>>>(data);
	DLOG(INFO) << "register worker (nid, lid) pairs: " << res;
	stat.t_data_deserial += tmr.elapseSd();
	for(auto& p : res){
		wm.registerID(p.first, p.second);
	}
	//rph.input(MType::CWorkers, info.source);
	suOnline.notify();
	sendReply(info, MType::CWorkers);
}

void Worker::handleStart(const std::string & data, const RPCInfo & info)
{
	suStart.notify();
}

void Worker::handlePause(const std::string & data, const RPCInfo & info)
{
	pauseTrain();
	sendReply(info, MType::CTrainPause);
}

void Worker::handleContinue(const std::string & data, const RPCInfo & info)
{
	resumeTrain();
	sendReply(info, MType::CTrainContinue);
}

void Worker::handleDeltaRequest(const std::string& data, const RPCInfo& info)
{
	requestingDelta = true;
	pauseTrain();
}

void Worker::handleLossRequest(const std::string& data, const RPCInfo& info)
{
	auto reqRatio = deserialize<pair<double, double>>(data);
	lossReqStart = static_cast<size_t>(reqRatio.first * pdh->size());
	lossReqCount = static_cast<size_t>(reqRatio.second * pdh->size());
	suLossReq.notify();
}

void Worker::handleProbeDone(const std::string& data, const RPCInfo& info)
{
	suProbeDone.notify();
}

void Worker::handleReset(const std::string& data, const RPCInfo& info)
{
	auto msg = deserialize<pair<int, vector<double>>>(data);
	iter = msg.first;
	Parameter p;
	p.set(move(msg.second));
	bufferParameter(p);
	pauseTrain();
	exitTrain = true;
}

void Worker::handleMetaConf(const std::string& data, const RPCInfo& info)
{
	pair<size_t, size_t> p = deserialize<pair<size_t, size_t>>(data);
	if(p.first != 0)
		localBatchSize = (this->*lbsFun)(p.first);
	if(p.second != 0)
		localReportSize = p.second;
	suConf.notify();
}

void Worker::handleImmediateControl(const std::string & data, const RPCInfo & info)
{
	int type = deserialize<int>(data);
	//const char* p = data.data() + sizeof(int);
	switch(type){
	case MType::CTerminate:
		handleTerminate(data.substr(sizeof(int)), info);
		break;
	}
}
void Worker::handleTerminate(const std::string & data, const RPCInfo & info)
{
	exitTrain = true;
	exitRun = true;
	pauseTrain(); // in case if the system is calculating delta
	suParam.notify(); // in case if the system just calculated a delta (is waiting for new parameter)
	sendReply(info, MType::CTerminate);
}
