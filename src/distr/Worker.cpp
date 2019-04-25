#include "Worker.h"
#include "common/Option.h"
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
	bfDeltaDpCount = 0;

	hasNewParam = false;
	allowTrain = true;
	exitTrain = false;
}

void Worker::init(const Option* opt, const size_t lid)
{
	this->opt = opt;
	nWorker = opt->nw;
	localID = lid;
	trainer = TrainerFactory::generate(opt->optimizer, opt->optimizerParam);
	LOG_IF(trainer == nullptr, FATAL) << "Trainer is not set correctly";
	ln = opt->logIter;
	logName = "W"+to_string(localID);
	setLogThreadName(logName);
	model.init(opt->algorighm, opt->algParam);

	if(opt->mode == "bsp"){
		bspInit();
	} else if(opt->mode == "tap"){
		tapInit();
	} else if(opt->mode == "ssp"){
		sspInit();
	} else if(opt->mode == "sap"){
		sapInit();
	} else if(opt->mode == "fsp"){
		fspInit();
	} else if(opt->mode == "aap"){
		aapInit();
	}
}

void Worker::bindDataset(const DataHolder* pdh)
{
	VLOG(1) << "Bind dataset with " << pdh->size() << " data points";
	trainer->bindDataset(pdh);
	// separated the mini-batch among all workers
	localBatchSize = opt->batchSize / nWorker;
	if(opt->batchSize % nWorker > localID)
		++localBatchSize;
	if(localBatchSize <= 0)
		localBatchSize = 1;
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

	DLOG(INFO) << "start training with mode: " << opt->mode << ", local batch size: " << localBatchSize;
	iter = 1;
	iterParam = 1;
	//try{
	//	generalProcess();
	//} catch(exception& e){
	//	LOG(FATAL) << e.what();
	//}
	if(opt->mode == "bsp"){
		bspProcess();
	} else if(opt->mode == "tap"){
		tapProcess();
	} else if(opt->mode == "ssp"){
		sspProcess();
	} else if(opt->mode == "sap"){
		sapProcess();
	} else if(opt->mode == "fsp"){
		fspProcess();
	} else if(opt->mode == "aap"){
		aapProcess();
	}

	DLOG(INFO) << "finish training";
	sendClosed();
	finishStat();
	delete trainer;
	trainer = nullptr;
	showStat();
	stopMsgLoop();
}

Worker::callback_t Worker::localCBBinder(
	void (Worker::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
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

	//addRPHAnySU(MType::DParameter, suParam);
	addRPHAnySU(MType::CDataset, suDatasetInfo);
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

void Worker::sendDelta(std::vector<double>& delta, const size_t cnt)
{
	DVLOG(3) << "send delta: " << delta;
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	net->send(masterNID, MType::DDelta, make_pair(move(cnt), move(delta)));
	++stat.n_dlt_send;
}

void Worker::initializeParameter()
{
	if(model.getKernel()->needInitParameterByData()){
		// model.param is set in trainer->ready()
		net->send(masterNID, MType::DParameter, model.getParameter().weights);
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

void Worker::pauseTrain()
{
	allowTrain = false;
}

void Worker::resumeTrain()
{
	allowTrain = true;
}

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

void Worker::handleImmediateControl(const std::string & data, const RPCInfo & info)
{
	int type = deserialize<int>(data);
	//const char* p = data.data() + sizeof(int);
	LOG(INFO) << "immediate: " << type;
	switch(type){
	case MType::CTerminate:
		handleTerminate(data.substr(sizeof(int)), info);
		break;
	}
}
void Worker::handleTerminate(const std::string & data, const RPCInfo & info)
{
	LOG(INFO) << "handle terminate";
	exitTrain = true;
	pauseTrain(); // in case if the system is calculating delta
	suParam.notify(); // in case if the system just calculated a delta (is waiting for new parameter)
	sendReply(info, MType::CTerminate);
}
