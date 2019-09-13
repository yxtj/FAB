#include "Worker.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"
#include <random>
#include <functional>

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
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false);
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
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false);
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
	requestingDelta = false;
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterPap));
}

void Worker::papProcess()
{
	// initialize speed adjustment generator
	const double speedSlowFactor = conf->adjustSpeedHetero ? conf->speedHeterogenerity[localID] : 0.0;
	mt19937 gen(conf->seed + 123 + localID);
	exponential_distribution<double> distExp;
	normal_distribution<double> distNorm;
	uniform_real_distribution<double> distUni;
	function<double()> speedRandomFun = [](){ return 0.0; };
	if(conf->adjustSpeedRandom){
		const vector<string>& param = conf->speedRandomParam;
		if(param[0] == "exp"){
			distExp.param(typename exponential_distribution<double>::param_type(1));
			speedRandomFun = [&](){ return distExp(gen); };
		} else if(param[0] == "norm"){
			distNorm.param(normal_distribution<double>::param_type(stod(param[1]), stod(param[2])));
			speedRandomFun = [&](){ return distNorm(gen); };
		} else if(param[0] == "uni"){
			//distUni = uniform_real_distribution<double>(stod(param[1]), stod(param[2]));
			distUni.param(uniform_real_distribution<double>::param_type(stod(param[1]), stod(param[2])));
			speedRandomFun = [&](){ return distUni(gen); };
		}
	}

	double t_data = 0.0, t_delta = 0.0, t_report = 0.0;
	size_t n_delta = 0, n_report = 0;

	while(!exitTrain){
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		size_t n_used = 0;
		t_data = 0.0;
		size_t left = localReportSize;
		double dly = speedSlowFactor + speedRandomFun();
		clearDelta();
		while(exitTrain == false && !requestingDelta){
			tmr.restart(); //
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			//tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, remainCnt, dly, -1);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			VLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points";
			if(dr.n_scanned != 0){
				accumulateDelta(dr.delta);
			}
			auto t= tmr.elapseSd();
			t_data += t;
			stat.t_dlt_calc += t;
			if(left == 0){
				tmr.restart();
				vector<double> report = { static_cast<double>(n_used),
					t_data / n_used, t_delta / n_delta, t_report / n_report };
				// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending
				VLOG_EVERY_N(ln, 2) << "  send report";
				sendReport(report);
				++n_report;
				t_report += tmr.elapseSd();
				left = localBatchSize;
			}
			if(hasNewParam){
				tmr.restart();
				applyBufferParameter();
				stat.t_par_calc += tmr.elapseSd();
			}
		}
		if(requestingDelta){
			tmr.restart();
			if(trainer->needAveragedDelta())
				averageDelta(n_used);
			stat.t_dlt_calc += tmr.elapseSd();
			VLOG_EVERY_N(ln, 2) << "  send delta";
			tmr.restart();
			sendDelta(bfDelta, n_used);
			requestingDelta = false;
			++n_delta;
			t_delta += tmr.elapseSd();
		}
		++iter;
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
	handleParameterAap(data, info);
}
