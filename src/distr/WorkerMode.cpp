#include "Worker.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"
#include <random>

using namespace std;

// ---- bulk synchronous parallel

void Worker::bspInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::bspProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t left = localBatchSize;
		size_t n_used = 0;
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
		} while(left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, localBatchSize);
		if(exitTrain == true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();
		if(exitTrain == true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

// ---- typical asynchronous parallel

void Worker::tapInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::tapProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
		} while(left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, localBatchSize);
		if(exitTrain == true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();
		if(exitTrain == true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

// ---- staleness synchronous parallel

void Worker::sspInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterSsp));
}

void Worker::sspProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t left = localBatchSize;
		size_t n_used = 0;
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
		} while(left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, localBatchSize);
		if(exitTrain == true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		while(!exitTrain && iter - iterParam > conf->staleGap){
			waitParameter();
		}
		if(exitTrain == true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

// ---- staleness asynchronous parallel

void Worker::sapInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterSsp));
}

void Worker::sapProcess()
{
	localBatchSize = trainer->pd->size();
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
		} while(left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, localBatchSize);
		if(exitTrain == true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		while(!exitTrain && iter - iterParam > conf->staleGap){
			waitParameter();
		}
		if(exitTrain == true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

// ---- flexible synchronous parallel

void Worker::fspInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterFsp));
}

void Worker::fspProcess()
{
	localBatchSize = trainer->pd->size();
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t left = trainer->pd->size();
		size_t n_used = 0;
		clearDelta();
		while(exitTrain == false && allowTrain && left != 0) {
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
		}
		// wait until allowTrain is set to false
		while(allowTrain == true)
			sleep();
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  calculate delta with " << n_used << " data points";
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used);
		if(exitTrain == true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter(); // resumeTrain() by handleParameterFsb()
		if(exitTrain == true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

// ---- aggressive asynchronous parallel

void Worker::aapInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterAap));
}

void Worker::aapProcess()
{
	// require different handleParameter -> handleParameterFab
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		clearDelta();
		bool newBatch = true;
		while(!exitTrain && left != 0){
			tmr.restart();
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			//DVLOG(3) <<"tmp: "<< tmp;
			VLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points, left: " << left;
			if(dr.n_scanned != 0){
				accumulateDelta(dr.delta);
			}
			stat.t_dlt_calc += tmr.elapseSd();
			tmr.restart();
			if(conf->aapWait && iter != 1 && newBatch){
				waitParameter();
				stat.t_par_wait += tmr.elapseSd();
				tmr.restart();
			}
			applyBufferParameter();
			stat.t_par_calc += tmr.elapseSd();
			newBatch = false;
		}
		tmr.restart();
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		sendDelta(bfDelta, n_used);
		//if(conf->aapWait)
		//	waitParameter();
		//stat.t_par_wait += tmr.elapseSd();
		++iter;
	}
}

// ---- progressive asynchronous parallel

void Worker::papInit()
{
	reqDelta = false;
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterPap));
	regDSPProcess(MType::DRDelta, localCBBinder(&Worker::handleDeltaRequest));
}

void Worker::papProcess()
{
	double sendT = 0;
	mt19937 gen(conf->seed + localID);
	double lambda = stod(conf->speedRandomParam[3]);
	exponential_distribution<double> distribution(lambda);

	double dly = lambda > 0 ? distribution(gen) : 0; /// random seed......
	dly = dly > 1 || dly < 0.1 ? 0 : dly * range;
	// size_t remaincnt = localBatchSize;
	while(!exitTrain){
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		size_t cnt = 0;
		int remainCnt = reportSize;
		if(opt->mode.find("pasp1") != std::string::npos)
			remainCnt = reportSize - curCnt;

		Trainer::DeltaResult dr = trainer->batchDelta(dataPointer, left, false);
		//tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, remainCnt, dly, -1);
		updatePointer(dr.n_scanned, dr.n_reported);

		copyDelta(bufferDeltaExt, bfDelta);
		curCnt += cnt;
		curCalT += tmr.elapseSd();
		stat.t_dlt_calc += tmr.elapseSd();

		if(!allowTrain){
			VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
			tmr.restart();
			if(reqDelta){
				DVLOG(3) << "send delta pasp: " << curCnt << "; " << bfDelta.size() << "; " << bfDelta;
				sendDelta(bufferDeltaExt, curCnt);
				reqDelta = false;
				++iter;
				VLOG_IF(iter < 5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT
					<< "; unit dp " << curCnt << " : " << curCalT / curCnt << " dly: " << dly
					<< " sendT: " << sendT / curCnt;
				bufferDeltaExt.clear();
				curCnt = 0;
				curCalT = 0;
			}
			if(hasNewParam){
				tmr.restart();
				applyBufferParameter();
				hasNewParam = false;
				double updateParamT = tmr.elapseSd();
				stat.t_par_calc += tmr.elapseSd();
				VLOG_IF(iter < 5 && (localID < 3), 1) << "iter " << iter << " interrupt CALT: " << curCalT
					<< "; unit dp " << cnt << " : " << curCalT / cnt << " dly: " << dly
					<< " ParamT: " << updateParamT;

				dly = lambda > 0 ? distribution(gen) : 0;
				dly = dly > 1 || dly < 0.1 ? 0 : dly * range;
			}
			allowTrain = true;

		} else{
			tmr.restart();
			// VLOG_EVERY_N(ln, 2) << "  send delta";
			if(opt->mode.find("pasp1") != std::string::npos ||
				opt->mode.find("pasp5") != std::string::npos){
				sendDelta(bfDelta, curCnt);
				VLOG_IF(iter < 5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT
					<< "; unit dp " << curCnt << " : " << curCalT / curCnt << " dly: " << dly
					<< " sendT: " << sendT / curCnt;
				bfDelta.clear();
				curCnt = 0;
			} else {// if (opt->mode.find("pasp") !=std::string::npos)
				sendReport(cnt);
			}
			if(opt->mode.find("pasp2") != std::string::npos)
				model.accumulateParameter(bfDelta, factorDelta);

			sendT += tmr.elapseSd();
			stat.t_dlt_send += tmr.elapseSd();
			// remaincnt = localBatchSize;
		}
		if(exitTrain == true){
			break;
		}
	}
}


// ---- handlers ----

void Worker::handleParameter(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}

void Worker::handleParameterSsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	++iterParam;
	//sendReply(info);
	++stat.n_par_recv;
}

void Worker::handleParameterFsp(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	// resume training
	resumeTrain();
	++stat.n_par_recv;
}

void Worker::handleParameterAap(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	// break the trainning and apply the received parameter (in main thread)
	pauseTrain();
	//applyBufferParameter();
	++stat.n_par_recv;
}

void Worker::handleParameterPap(const std::string& data, const RPCInfo& info)
{
}
