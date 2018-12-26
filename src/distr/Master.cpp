#include "Master.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
using namespace std;

Master::Master() : Runner() {
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	trainer.bindModel(&model);
	factorDelta = 1.0;
	nx = 0;
	iter = 0;
	nUpdate = 0;
	lastArchIter = 0;
	tmrArch.restart();

	suOnline.reset();
	suWorker.reset();
	suXLength.reset();
	suParam.reset();
	suDeltaAny.reset();
	suDeltaAll.reset();
	suTPause.reset();
	suTContinue.reset();
	suAllClosed.reset();
}

void Master::init(const Option* opt, const size_t lid)
{
	this->opt = opt;
	nWorker = opt->nw;
	trainer.setRate(opt->lrate);
	localID = lid;
	ln = opt->logIter;
	logName = "M";
	setLogThreadName(logName);
	if(opt->mode == "sync"){
		syncInit();
	} else if(opt->mode == "async"){
		asyncInit();
	} else if(opt->mode == "fsb"){
		fsbInit();
	} else if(opt->mode == "fab"){
		fabInit();
	}
}

void Master::run()
{
	registerHandlers();
	startMsgLoop(logName+"-MSG");
	
	LOG(INFO) << "Wait online messages";
	suOnline.wait();
	LOG(INFO) << "Send worker list";
	broadcastWorkerList();
	net->startMeasureBW(opt->tcTime);
	LOG(INFO)<<"Waiting x-length to initialize parameters";
	initializeParameter();
	LOG(INFO) << "Got x-length = " << nx;
	if(!opt->fnOutput.empty()){
		foutput.open(opt->fnOutput);
		LOG_IF(foutput.fail(), FATAL) << "Cannot write to file: " << opt->fnOutput;
	}
	iter = 0;
	tmrTrain.restart();
	archiveProgress(true);
	LOG(INFO) << "Broadcasting initial parameter";
	broadcastParameter();

	LOG(INFO)<<"Start traning with mode: "<<opt->mode;
	//tmrTrain.restart();
	iter = 1;
	if(opt->mode == "sync"){
		syncProcess();
	} else if(opt->mode == "async"){
		asyncProcess();
	} else if(opt->mode == "fsb"){
		fsbProcess();
	} else if(opt->mode == "fab"){
		fabProcess();
	}
	double t = tmrTrain.elapseSd();
	LOG(INFO) << "Finish training. Time cost: " << t << ". Iterations: " << iter
		<< ". Average iteration time: " << t / iter;

	net->finishMeasureBW();
	broadcastSignalTerminate();
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaTail));
	foutput.close();
	dumpBwUsage(net->getBWUsage());
	DLOG(INFO) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
	suAllClosed.wait();
	stopMsgLoop();
}

Master::callback_t Master::localCBBinder(
	void (Master::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}

void Master::syncInit()
{
	factorDelta = 1.0 / nWorker;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::syncProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter;
		waitDeltaFromAll();
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		archiveProgress();
		//waitParameterConfirmed();
		++iter;
	}
}

void Master::asyncInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaAsync));
}

void Master::asyncProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
			newIter = false;
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		size_t p = nUpdate / nWorker + 1;
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

void Master::fsbInit()
{
	factorDelta = 1.0 / nWorker;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::fsbProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter;
		waitDeltaFromAny();
		VLOG_EVERY_N(ln, 2) << "  Broadcast pause signal";
		broadcastSignalPause();
		VLOG_EVERY_N(ln, 2) << "  Waiting for all deltas";
		waitDeltaFromAll();
		suDeltaAny.reset();
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		//waitParameterConfirmed();
		VLOG_EVERY_N(ln, 2) << "  Broadcast continue signal";
		//broadcastSignalContinue();
		archiveProgress();
		++iter;
	}
}

void Master::fabInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaFab));
}

void Master::fabProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;// << ", msg waiting: " << driver.queSize() << ", update: " << nUpdate;
			DVLOG_EVERY_N(ln / 10, 1) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
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
		broadcastParameter();
		size_t p = nUpdate / nWorker + 1;
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

void Master::registerHandlers()
{
	regDSPProcess(MType::CReply, localCBBinder(&Master::handleReply));
	regDSPProcess(MType::COnline, localCBBinder(&Master::handleOnline));
	regDSPProcess(MType::CXLength, localCBBinder(&Master::handleXLength));
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta)); // for sync and fsb
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaAsync)); // for async

	addRPHEachSU(MType::COnline, suOnline);
	addRPHEachSU(MType::CWorkers, suWorker);
	addRPHEachSU(MType::CXLength, suXLength);
	addRPHEachSU(MType::DParameter, suParam);
	addRPHEachSU(MType::CTrainPause, suTPause);
	addRPHEachSU(MType::CTrainContinue, suTContinue);
	addRPHEachSU(MType::CTerminate, suAllClosed);
	addRPHAnySU(typeDDeltaAny, suDeltaAny);
	addRPHEachSU(typeDDeltaAll, suDeltaAll);
}

void Master::bindDataset(const DataHolder* pdh)
{
	trainer.bindDataset(pdh);
}

void Master::applyDelta(std::vector<double>& delta, const int source)
{
	DVLOG(3) << "apply delta from " << source << " : " << delta
		<< "\nonto: " << model.getParameter().weights;
	model.accumulateParameter(delta, factorDelta);
}

bool Master::terminateCheck()
{
	return (iter >= opt->tcIter)
		|| (tmrTrain.elapseSd() > opt->tcTime);
}

void Master::initializeParameter()
{
	suXLength.wait();
	suXLength.reset();
	model.init(opt->algorighm, nx, opt->algParam, 0.01);
}

void Master::sendParameter(const int target)
{
	DVLOG(3) << "send parameter to " << target << " with: " << model.getParameter().weights;
	net->send(wm.lid2nid(target), MType::DParameter, model.getParameter().weights);
}

void Master::broadcastParameter()
{
	DVLOG(3) << "broad parameter: " << model.getParameter().weights;
	net->broadcast(MType::DParameter, model.getParameter().weights);
}

void Master::waitParameterConfirmed()
{
	suParam.wait();
	suParam.reset();
}

bool Master::needArchive()
{
	if(!foutput.is_open())
		return false;
	if(iter - lastArchIter >= opt->arvIter
		|| tmrArch.elapseSd() >= opt->arvTime)
	{
		lastArchIter = iter;
		tmrArch.restart();
		return true;
	}
	return false;
}

void Master::archiveProgress(const bool force)
{
	if(!force && !needArchive())
		return;
	foutput << iter << "," << tmrTrain.elapseSd();
	for(auto& v : model.getParameter().weights){
		foutput << "," << v;
	}
	foutput <<"\n";
}

void Master::broadcastWorkerList()
{
	vector<pair<int, int>> temp = wm.list();
	net->broadcast(MType::CWorkers, temp);
	suWorker.wait();
}

void Master::broadcastSignalPause()
{
	net->broadcast(MType::CTrainPause, "");
	suTPause.wait();
	suTPause.reset();
}

void Master::broadcastSignalContinue()
{
	net->broadcast(MType::CTrainContinue, "");
	suTContinue.wait();
	suTContinue.reset();
}

void Master::broadcastSignalTerminate()
{
	net->broadcast(MType::CTerminate, "");
}

void Master::waitDeltaFromAny(){
	suDeltaAny.wait();
}

void Master::waitDeltaFromAll(){
	suDeltaAll.wait();
	suDeltaAll.reset();
}

void Master::gatherDelta()
{
	suDeltaAll.reset();
	net->broadcast(MType::DRDelta, "");
	suDeltaAll.wait();
}

void Master::handleReply(const std::string & data, const RPCInfo & info)
{
	int type = deserialize<int>(data);
	int source = wm.nid2lid(info.source);
	rph.input(type, source);
}

void Master::handleOnline(const std::string & data, const RPCInfo & info)
{
	auto lid = deserialize<int>(data);
	wm.registerID(info.source, lid);
	rph.input(MType::COnline, lid);
	sendReply(info);
}

void Master::handleXLength(const std::string& data, const RPCInfo& info){
	size_t d = deserialize<size_t>(data);
	int source = wm.nid2lid(info.source);
	if(nx == 0){
		nx = d;
	} else if(nx != d){
		LOG(FATAL)<<"dataset on "<<source<<" does not match with others";
	}
	rph.input(MType::CXLength, source);
	sendReply(info);
}

void Master::handleDelta(const std::string & data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
}

void Master::handleDeltaAsync(const std::string & data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	++nUpdate;
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaFab(const std::string & data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	++nUpdate;
	//static vector<int> cnt(nWorker, 0);
	//++cnt[s];
	//VLOG_EVERY_N(ln/10, 1) << "Update: " << nUpdate << " rsp: " << cnt << " r-pkg: " << net->stat_recv_pkg;
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	if(opt->fabWait)
		sendReply(info);
	// broadcast new parameter in main thread
}

void Master::handleDeltaTail(const std::string & data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
}
