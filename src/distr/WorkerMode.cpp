#include "Worker.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"
#include <random>
#include <functional>

using namespace std;

// ---- general probe mode

void Worker::probeModeInit()
{
	(this->*initFun)();
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterProbe));
}

void Worker::probeModeProcess()
{
	size_t probeNeededPoint = static_cast<size_t>(conf->probeRatio * pdh->size());
	double loss = calcLoss(0, probeNeededPoint);
	sendLoss({loss});
	while(!exitRun){
		LOG(INFO) << "waiting for new configuration";
		suConf.wait_n_reset();
		(this->*processFun)();
		applyBufferParameter();
	}
}

void Worker::handleParameterProbe(const std::string& data, const RPCInfo& info)
{
	(this->*paramFun)(data, info);
}

// ---- bulk synchronous parallel

void Worker::bspInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::bspProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;
		Timer tmr;
		size_t left = localBatchSize;
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		//VLOG_EVERY_N(ln, 2) << "dly=" << dly;
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
		} while(!exitTrain && left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used, loss);
		if(exitTrain == true){
			break;
		}
		DVLOG_EVERY_N(ln, 2) << "  wait for new parameter";
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
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
		} while(!exitTrain && left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used, loss);
		if(exitTrain == true){
			break;
		}
		DVLOG_EVERY_N(ln, 2) << "  wait for new parameter";
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
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;
		Timer tmr;
		size_t left = localBatchSize;
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
		} while(!exitTrain && left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used, loss);
		if(exitTrain == true){
			break;
		}
		DVLOG_EVERY_N(ln, 2) << "  wait for new parameter";
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
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		do{
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
		} while(!exitTrain && left > 0);
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used, loss);
		if(exitTrain == true){
			break;
		}
		DVLOG_EVERY_N(ln, 2) << "  wait for new parameter";
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
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;
		Timer tmr;
		size_t left = trainer->pd->size();
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		while(allowTrain && !exitTrain && left != 0) {
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			accumulateDelta(dr.delta);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
		}
		// wait until allowTrain is set to false
		while(allowTrain == true)
			sleep();
		if(trainer->needAveragedDelta())
			averageDelta(n_used);
		stat.t_dlt_calc += tmr.elapseSd();
		DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << n_used << " data points";
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, n_used, loss);
		if(exitTrain == true){
			break;
		}
		DVLOG_EVERY_N(ln, 2) << "  wait for new parameter";
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
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		size_t left = localBatchSize;
		// make the reporting time more even
		if(iter == 1)
			left += localBatchSize * localID / nWorker;
		size_t n_used = 0;
		double loss = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		bool newBatch = true;
		while(!exitTrain && left != 0){
			tmr.restart();
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
			//DVLOG(3) <<"tmp: "<< tmp;
			DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points, left: " << left;
			if(dr.n_reported != 0){
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
		DVLOG_EVERY_N(ln, 2) << "  send delta";
		sendDelta(bfDelta, n_used, loss);
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
	if(conf->papOnlineProbeVersion == 1)
		papOnlineProbe1();
	else if(conf->papOnlineProbeVersion == 2)
		papOnlineProbe2();
	else if(conf->papOnlineProbeVersion == 3)
		papOnlineProbe3();
	else if(conf->papOnlineProbeVersion == 4)
		papOnlineProbe4();
	else if(conf->papOnlineProbeVersion == 5)
		papOnlineProbeBenchmark();

	double t_data = 0.0, t_delta = 0.0, t_report = 0.0;
	size_t n_delta = 0, n_report = 0;

	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		while(exitTrain == false && !requestingDelta){
			tmr.restart(); //
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
			DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points";
			if(dr.n_reported!= 0){
				accumulateDelta(dr.delta);
			}
			auto t= tmr.elapseSd();
			t_data += t;
			stat.t_dlt_calc += t;
			if(left == 0){
				tmr.restart();
				double avgtd = n_delta == 0 ? 0 : t_delta / n_delta;
				double avgtu = n_updParam == 0 ? 0 : t_updParam / n_updParam;
				double avgtr = n_report == 0 ? 0 : t_report / n_report;
				vector<double> report = { static_cast<double>(n_used - n_used_since_report),
					t_data / n_used, avgtd + avgtu, avgtr, loss - loss_since_report };
				// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
				DVLOG_EVERY_N(ln, 2) << "  send report: " << report;
				n_used_since_report = n_used;
				loss_since_report = loss;
				sendReport(report);
				++n_report;
				t_report += tmr.elapseSd();
				left = localReportSize;
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
			DVLOG_EVERY_N(ln, 2) << "  send delta";
			tmr.restart();
			sendDelta(bfDelta, n_used, loss);
			requestingDelta = false;
			++n_delta;
			t_delta += tmr.elapseSd();
		}
		++iter;
	}
}

void Worker::papOnlineProbe1()
{
	Timer tmr;
	double loss = calcLoss(0, localBatchSize);
	double t1 = tmr.elapseSd();
	sendLoss({loss});
	double t2 = tmr.elapseSd() - t1;
	// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss, time-per-new-parameter
	vector<double> report = { 0.0, t1 / localBatchSize, t2, t2, 0.0, 0.0 };
	sendReport(report);

	// the rest part is exactly the same as the normal training part,
	// so the working floww directly moves to the normal logic
}

void Worker::papOnlineProbe2()
{
	double t_data = 0.0, t_delta = 0.0, t_report = 0.0;
	size_t n_delta = 0, n_report = 0;
	size_t prevStart = 0;
	Parameter prevParam = model.getParameter();
	size_t n_probe = 0;

	while(!exitTrain && !suProbeDone.ready()){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		while(exitTrain == false && !requestingDelta && !suLossReq.ready() && !suProbeDone.ready()){
			tmr.restart(); //
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
			DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points";
			if(dr.n_reported!= 0){
				accumulateDelta(dr.delta);
			}
			auto t= tmr.elapseSd();
			t_data += t;
			stat.t_dlt_calc += t;
			if(left == 0){
				tmr.restart();
				double avgtd = n_delta == 0 ? 0 : t_delta / n_delta;
				double avgtu = n_updParam == 0 ? 0 : t_updParam / n_updParam;
				double avgtr = n_report == 0 ? 0 : t_report / n_report;
				vector<double> report = { static_cast<double>(n_used - n_used_since_report),
					t_data / n_used, avgtd + avgtu, avgtr, loss - loss_since_report };
				// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
				DVLOG_EVERY_N(ln, 2) << "  send report: " << report;
				n_used_since_report = n_used;
				loss_since_report = loss;
				sendReport(report);
				++n_report;
				t_report += tmr.elapseSd();
				left = localReportSize;
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
			DVLOG_EVERY_N(ln, 2) << "  send delta";
			tmr.restart();
			sendDelta(bfDelta, n_used, loss);
			requestingDelta = false;
			++n_delta;
			t_delta += tmr.elapseSd();
			n_probe += n_used;
		}
		// send init loss for current probe
		if (suLossReq.ready()) {
			Timer tmr;
			// VLOG(2) << " calc loss 4 cur probe batch from " << prevStart << " for " << n_probe;
			lock_guard<mutex> lk(mParam); // lock prameter
			
			double L1 = calcLoss(prevStart, n_probe);
			double LB1_100 = calcLoss(0, 100);
			// double LB1_200 = calcLoss(0, 200);
			double LB1_500 = calcLoss(0, 500);
			Parameter curParam = model.getParameter();
			model.setParameter(prevParam);
			double L0 = calcLoss(prevStart, n_probe);
			double LB0_100 = calcLoss(0, 100);
			// double LB0_200 = calcLoss(0, 200);
			double LB0_500 = calcLoss(0, 500);
			// VLOG(2) << " send loss " << L0;
			sendLoss({L0, L1, LB0_100 - LB1_100, LB0_500 - LB1_500});

			prevParam = curParam;
			model.setParameter(curParam);
			VLOG(2) << "Probe recalculate L for lrs=" << localReportSize << " w/n " << n_probe
					<< "\ttime: " << tmr.elapseSd();
			prevStart = dataPointer;
			n_probe = 0;
			suLossReq.reset();
		}
		++iter;
	}
}

void Worker::papOnlineProbe3()
{}

void Worker::papOnlineProbe4()
{
	double t_data = 0.0, t_delta = 0.0, t_report = 0.0;
	size_t n_delta = 0, n_report = 0;
	size_t prevStart = 0;
	size_t n_probe = 0;

	while(!exitTrain && !suProbeDone.ready()){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = speedFactor.generate();
		clearDelta();
		while(exitTrain == false && !requestingDelta && !suLossReq.ready() && !suProbeDone.ready()){
			tmr.restart(); //
			resumeTrain();
			Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, left, false, dly);
			updatePointer(dr.n_scanned, dr.n_reported);
			left -= dr.n_scanned;
			n_used += dr.n_reported;
			loss += dr.loss;
			DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points";
			if(dr.n_reported != 0){
				accumulateDelta(dr.delta);
			}
			auto t = tmr.elapseSd();
			t_data += t;
			stat.t_dlt_calc += t;
			if(left == 0){
				tmr.restart();
				double avgtd = n_delta == 0 ? 0 : t_delta / n_delta;
				double avgtu = n_updParam == 0 ? 0 : t_updParam / n_updParam;
				double avgtr = n_report == 0 ? 0 : t_report / n_report;
				vector<double> report = { static_cast<double>(n_used - n_used_since_report),
					t_data / n_used, avgtd + avgtu, avgtr, loss - loss_since_report };
				// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
				DVLOG_EVERY_N(ln, 2) << "  send report: " << report;
				n_used_since_report = n_used;
				loss_since_report = loss;
				sendReport(report);
				++n_report;
				t_report += tmr.elapseSd();
				left = localReportSize;
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
			DVLOG_EVERY_N(ln, 2) << "  send delta";
			tmr.restart();
			sendDelta(bfDelta, n_used, loss);
			requestingDelta = false;
			++n_delta;
			t_delta += tmr.elapseSd();
			n_probe += n_used;
		}
		// send init loss for current probe
		if(suLossReq.ready()) {
			Timer tmr;
			double Ln = calcLoss(prevStart, n_probe);
			sendLoss({ Ln, 0.0, 0.0, 0.0 });
			VLOG(2) << "Probe loss for lrs=" << localReportSize << " from " << prevStart << " with " << n_probe
				<< " time: " << tmr.elapseSd() << " Ln=" << Ln << " unit-loss=" << Ln / n_probe;
			prevStart = dataPointer;
			n_probe = 0;
			suLossReq.reset();
		}
		++iter;
	}
}

void Worker::papOnlineProbeBenchmark()
{
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
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	pauseTrain();
	//applyBufferParameter();
	++stat.n_par_recv;
}
