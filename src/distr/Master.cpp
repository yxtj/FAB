#include "Master.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"
#include <future>

using namespace std;

Master::Master() : Runner() {
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	trainer.bindModel(&model);
	factorDelta = 1.0;
	nx = 0;
	ny = 0;
	nPoint = 0;
	iter = 0;
	nUpdate = 0;
	pie = nullptr;
	prs = nullptr;
	lastArchIter = 0;
	tmrArch.restart();
	doArchive = false;
	archDoing = false;

	suOnline.reset();
	suWorker.reset();
	suDatasetInfo.reset();
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
	nPointWorker.assign(nWorker, 0);
	trainer.setRate(opt->lrate);
	localID = lid;
	ln = opt->logIter;
	logName = "M";
	setLogThreadName(logName);
	model.init(opt->algorighm, opt->algParam, opt->seed);

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
		pie =IntervalEstimatorFactory::generate(opt->intervalParam, nWorker, nPoint);
		LOG_IF(pie == nullptr, FATAL) << "Fail to initialize interval estimator with parameter: " << opt->intervalParam;
	} else if(opt->mode == "aap"){
		aapInit();
		prs = ReceiverSelectorFactory::generate(opt->mcastParam, nWorker);
		LOG_IF(prs == nullptr, FATAL) << "Fail to initialize receiver selector with parameter: " << opt->mcastParam;
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
	LOG(INFO)<<"Waiting dataset info to initialize parameters";
	checkDataset();
	clearAccumulatedDelta();
	LOG(INFO) << "Got x-length: " << nx << ", y-length: " << ny << ", data points: " << nPoint;
	if(!opt->fnOutput.empty()){
		doArchive = true;
		archiver.init_write(opt->fnOutput, model.paramWidth(), false, opt->binary);
		LOG_IF(!archiver.valid(), FATAL) << "Cannot write to file: " << opt->fnOutput;
	}
	iter = 0;
	tmrTrain.restart();
	archiveProgress(true);
	LOG(INFO) << "Broadcasting initial parameter";
	broadcastParameter();

	LOG(INFO)<<"Start traning with mode: "<<opt->mode;
	//tmrTrain.restart();
	iter = 1;
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
	double t = tmrTrain.elapseSd();
	LOG(INFO) << "Finish training. Time cost: " << t << ". Iterations: " << iter
		<< ". Average iteration time: " << t / iter;

	broadcastSignalTerminate();
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaTail));
	archiver.close();
	delete pie;
	delete prs;
	DLOG(INFO) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
	finishStat();
	showStat();
	rph.deactivateType(MType::DDelta);
	suAllClosed.wait();
	stopMsgLoop();
}

Master::callback_t Master::localCBBinder(
	void (Master::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}


void Master::registerHandlers()
{
	regDSPProcess(MType::CReply, localCBBinder(&Master::handleReply));
	regDSPProcess(MType::COnline, localCBBinder(&Master::handleOnline));
	regDSPProcess(MType::CDataset, localCBBinder(&Master::handleDataset));
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta)); // for bsp and fsp
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaTap)); // for tap

	addRPHEachSU(MType::COnline, suOnline);
	addRPHEachSU(MType::CWorkers, suWorker);
	addRPHEachSU(MType::CDataset, suDatasetInfo);
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
	Timer tmr;
	DVLOG(3) << "apply delta from " << source << " : " << delta
		<< "\nonto: " << model.getParameter().weights;
	model.accumulateParameter(delta, factorDelta);
	stat.n_point += bfDeltaDpCount;
	stat.t_par_calc += tmr.elapseSd();
}

void Master::clearAccumulatedDelta()
{
	bfDelta.assign(model.paramWidth(), 0.0);
	bfDeltaDpCount = 0;
}

void Master::accumulateDelta(const std::vector<double>& delta, const size_t cnt)
{
	Timer tmr;
	for(size_t i = 0; i < delta.size(); ++i)
		bfDelta[i] += delta[i];
	bfDeltaDpCount += cnt;
	stat.t_dlt_calc += tmr.elapseSd();
}

void Master::applyDeltaNext(const int d)
{
	if(d == 0)
		return;
	int size = min<int>(d, static_cast<int>(bfDelta.size()) - 1);
	for(int i = 1; i <= size; ++i){
		if(!bfDeltaNext[i].empty()){
			applyDelta(bfDeltaNext[i], -1);
			stat.n_point += bfDeltaDpCountNext[i];
		}
	}
}

void Master::clearAccumulatedDeltaNext(const int d)
{
	// move slot[d+1] to bfDelta
	// move slot[d+2,...] to slot[1,...]
	Timer tmr;
	size_t bfSize = bfDeltaNext.size();
	// set bfDelta
	if(bfSize <= d + 1){ // no slot d+1
		//VLOG(1) << "no slot " << d;
		clearAccumulatedDelta();
		bfDeltaNext.resize(1);
		bfDeltaDpCountNext.resize(1);
	} else{
		//VLOG(1) << "move " << d << " " << bfSize << " " << iter << " " << deltaIter;
		bfDelta = move(bfDeltaNext[d+1]);
		bfDeltaDpCount = bfDeltaDpCountNext[d+1];
	}
	// set bfDeltaNext
	size_t p = 1;
	for(size_t i = d + 2; i < bfSize; ++i){
		bfDeltaNext[p] = move(bfDeltaNext[i]);
		bfDeltaDpCountNext[p] = bfDeltaDpCountNext[i];
		++p;
	}
	if(bfSize >= d + 2){
		bfDeltaNext.erase(bfDeltaNext.begin() + (bfSize - d - 1), bfDeltaNext.end());
		bfDeltaDpCountNext.erase(bfDeltaDpCountNext.begin() + (bfSize - d - 1), bfDeltaDpCountNext.end());
	}
	stat.t_dlt_calc += tmr.elapseSd();
}

void Master::shiftAccumulatedDeltaNext()
{
	// move slot[1] to bfDelta
	// move slot[2,...] to slot[1,...]
	Timer tmr;
	size_t bfSize = bfDeltaNext.size();
	if(bfSize <= 1){
		clearAccumulatedDelta();
	} else {
		bfDelta = move(bfDeltaNext[1]);
		bfDeltaDpCount = bfDeltaDpCountNext[1];
	}
	for(size_t i = 2; i < bfSize; ++i){
		bfDeltaNext[i - 1] = move(bfDeltaNext[i]);
		bfDeltaDpCountNext[i - 1] = bfDeltaDpCountNext[i];
	}
	if(bfSize >= 2){
		bfDeltaNext.pop_back();
		bfDeltaDpCountNext.pop_back();
	}
	stat.t_dlt_calc += tmr.elapseSd();
}

void Master::accumulateDeltaNext(const int d, const std::vector<double>& delta, const size_t cnt)
{
	Timer tmr;
	if(bfDeltaNext.size() <= d){
		bfDeltaNext.resize(d + 1, vector<double>(delta.size(), 0.0));
		bfDeltaDpCountNext.resize(d + 1, 0);
	} else if(bfDeltaNext[d].empty()){
		bfDeltaNext[d].assign(delta.size(), 0.0);
	}
	for(size_t i = 0; i < delta.size(); ++i)
		bfDeltaNext[d][i] += delta[i];
	bfDeltaDpCountNext[d] += cnt;
	stat.t_dlt_calc += tmr.elapseSd();
}

bool Master::terminateCheck()
{
	return (iter >= opt->tcIter)
		|| (tmrTrain.elapseSd() > opt->tcTime);
}

void Master::checkDataset()
{
	suDatasetInfo.wait_n_reset();
	model.checkData(nx, ny);
}

void Master::sendParameter(const int target)
{
	DVLOG(3) << "send parameter to " << target << " with: " << model.getParameter().weights;
	net->send(wm.lid2nid(target), MType::DParameter, model.getParameter().weights);
	++stat.n_par_send;
}

void Master::broadcastParameter()
{
	const auto& m = model.getParameter().weights;
	DVLOG(3) << "broadcast parameter: " << m;
	net->broadcast(MType::DParameter, m);
	stat.n_par_send += nWorker;
}

void Master::multicastParameter(const int source)
{
	const auto& m = model.getParameter().weights;
	vector<int> targets = prs->getTargets(source);
	DVLOG(3) << "multicast parameter: " << m << " to " << targets;
	for(int& v : targets)
		v=wm.lid2nid(v);
	net->multicast(targets, MType::DParameter, m);
	stat.n_par_send += targets.size();
}

void Master::waitParameterConfirmed()
{
	suParam.wait_n_reset();
}

bool Master::needArchive()
{
	if(!doArchive)
		return false;
	if(iter - lastArchIter >= opt->arvIter
		|| tmrArch.elapseSd() >= opt->arvTime)
	{
		return true;
	}
	return false;
}

void Master::archiveProgress(const bool force)
{
	if(!force && !needArchive())
		return;
	if(archDoing)
		return;
	archDoing = true;
	lastArchIter = iter;
	tmrArch.restart();
	++stat.n_archive;
	std::async(launch::async, [&](int iter, double time, Parameter param){
		Timer t;
		archiver.dump(iter, time, param);
//		archiver.dump(iter, tmrTrain.elapseSd(), model.getParameter());
		archDoing = false;
		stat.t_archive += t.elapseSd();
	}, iter, tmrTrain.elapseSd(), model.getParameter());
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
	suTPause.wait_n_reset();
}

void Master::broadcastSignalContinue()
{
	net->broadcast(MType::CTrainContinue, "");
	suTContinue.wait_n_reset();
}

void Master::broadcastSignalTerminate()
{
	net->broadcast(MType::CTerminate, "");
}

void Master::waitDeltaFromAny(){
	suDeltaAny.wait_n_reset();
}

void Master::waitDeltaFromAll(){
	suDeltaAll.wait_n_reset();
}

void Master::gatherDelta()
{
	suDeltaAll.reset();
	net->broadcast(MType::DRDelta, "");
	suDeltaAll.wait();
}

void Master::handleReply(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	int type = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int source = wm.nid2lid(info.source);
	rph.input(type, source);
}

void Master::handleOnline(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto lid = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	wm.registerID(info.source, lid);
	rph.input(MType::COnline, lid);
	sendReply(info);
}

void Master::handleDataset(const std::string& data, const RPCInfo& info){
	Timer tmr;
	size_t tnx, tny, tnp;
	tie(tnx, tny, tnp) = deserialize<tuple<size_t, size_t, size_t>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int source = wm.nid2lid(info.source);
	bool flag = false;
	if(nx == 0){
		nx = tnx;
	} else if(nx != tnx){
		flag = true;
	}
	if(ny == 0){
		ny = tny;
	} else if(ny != tny){
		flag = true;
	}
	LOG_IF(flag, FATAL) << "dataset on " << source << " does not match with others."
		<< " X-match: " << (nx == tnx) << ", Y-match: " << (ny == tny);
	nPointWorker[source] = tnp;
	nPoint += tnp;
	rph.input(MType::CDataset, source);
	sendReply(info);
}

void Master::handleDeltaTail(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto deltaMsg = deserialize<pair<size_t, vector<double>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	applyDelta(deltaMsg.second, s);
	++stat.n_dlt_recv;
}
