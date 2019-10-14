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
	size_t probeNeededPoint = static_cast<size_t>(
		(conf->probeLossFull ? 1.0 : conf->probeRatio) * pdh->size());
	double loss = calcLoss(0, probeNeededPoint);
	LOG(INFO) << "send initialize loss: " << loss;
	sendLoss(loss);
	while(!suProbeDone.ready()){
		LOG(INFO) << "waiting for new configuration";
		suConf.wait_n_reset();
		if(suProbeDone.ready())
			break;
		VLOG(2) << "probe local lbs=" << localBatchSize << " lrs=" << localReportSize;
		clearDelta();
		exitTrain = false;
		allowTrain = true;
		(this->*processFun)(); // stop by Reset message via setting exitTrain=true
		LOG(INFO) << "finish one probe";
		suLossReq.wait_n_reset();
		double loss = calcLoss(0, probeNeededPoint);
		LOG(INFO) << "send loss:" << loss;
		sendLoss(loss);
		applyBufferParameter(); // reset to the initialing parameter
	}
	LOG(INFO) << "probe done";
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
		double dly = getSpeedFactor();
		VLOG_EVERY_N(ln, 2) << "  dly=" << dly;
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
		double dly = getSpeedFactor();
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
		double dly = getSpeedFactor();
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
		double dly = getSpeedFactor();
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
		double dly = getSpeedFactor();
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
		double dly = getSpeedFactor();
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
	if(localReportSize > localBatchSize)
		localReportSize = localBatchSize / 2;
	if(localReportSize == 0)
		localReportSize = 1;
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterPap));
	LOG(INFO) << "lbs=" << localBatchSize << "\tlrs=" << localReportSize;
}

void Worker::papProcess()
{
	LOG(INFO) << "lbs=" << localBatchSize << "\tlrs=" << localReportSize;
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
	else if(conf->papOnlineProbeVersion == 9)
		papOnlineProbeFile();

	double t_data = 0.0; // data point processing time of current iteration

	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = getSpeedFactor();
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
				double avgtd = stat.n_dlt_send == 0 ? 0 : stat.t_dlt_send / stat.n_dlt_send;
				double avgtu = stat.t_par_calc == 0 ? 0 : stat.t_par_calc / stat.n_par_recv;
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
			stat.t_dlt_send += tmr.elapseSd();
		}
		++iter;
	}
}

void Worker::papOnlineProbe1()
{
	Timer tmr;
	double loss = calcLoss(0, localBatchSize);
	double t1 = tmr.elapseSd();
	sendLoss(loss);
	double t2 = tmr.elapseSd() - t1;
	// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss, time-per-new-parameter
	vector<double> report = { 0.0, t1 / localBatchSize, t2, t2, 0.0, 0.0 };
	sendReport(report);

	// the rest part is exactly the same as the normal training part,
	// so the working floww directly moves to the normal logic
}

void Worker::papOnlineProbe2()
{
	double t_calcLoss = 0.0;
	size_t prevStart = 0;
	Parameter prevParam = model.getParameter();
	size_t n_probe = 0;

	while(!exitTrain && !suProbeDone.ready()){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		double t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = getSpeedFactor();
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
				double avgtd = stat.n_dlt_send == 0 ? 0 : stat.t_dlt_send / stat.n_dlt_send;
				double avgtu = stat.t_par_calc == 0 ? 0 : stat.t_par_calc / stat.n_par_recv;
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
			stat.t_dlt_send += tmr.elapseSd();
			n_probe += n_used;
		}
		// send init loss for current probe
		if (suLossReq.ready()) {
			Timer tmr;
			// VLOG(2) << " calc loss 4 cur probe batch from " << prevStart << " for " << n_probe;
			lock_guard<mutex> lk(mParam); // lock prameter
			Parameter curParam = model.getParameter();
			model.setParameter(move(prevParam));
			double L0 = calcLoss(prevStart, n_probe);
			sendLoss(L0);
			model.setParameter(curParam);
			prevParam = move(curParam);
			t_calcLoss += tmr.elapseSd();
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
	size_t prevStart = 0;
	size_t n_probe = 0;

	while(!exitTrain && !suProbeDone.ready()){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		double t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = getSpeedFactor();
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
				double avgtd = stat.n_dlt_send == 0 ? 0 : stat.t_dlt_send / stat.n_dlt_send;
				double avgtu = stat.t_par_calc == 0 ? 0 : stat.t_par_calc / stat.n_par_recv;
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
			stat.t_dlt_send += tmr.elapseSd();
			n_probe += n_used;
		}
		// send init loss for current probe
		if(suLossReq.ready()) {
			Timer tmr;
			double Ln = calcLoss(prevStart, n_probe);
			sendLoss(Ln);
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
	double t_calcLoss = 0.0;
	size_t prevStart = 0;
	Parameter prevParam = model.getParameter();
	size_t n_probe = 0, n_bench = 100;
	double LB = calcLoss(0, n_bench) / n_bench;
	sendLoss(LB); // send unit init Loss

	while(!exitTrain && !suProbeDone.ready()){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter;// << ". msg waiting: " << driver.queSize();
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		double t_data = 0.0;
		size_t left = localReportSize;
		size_t n_used = 0, n_used_since_report = 0;
		double loss = 0.0, loss_since_report = 0.0;
		double dly = getSpeedFactor();
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
				double avgtd = stat.n_dlt_send == 0 ? 0 : stat.t_dlt_send / stat.n_dlt_send;
				double avgtu = stat.t_par_calc == 0 ? 0 : stat.t_par_calc / stat.n_par_recv;
				double avgtr = n_report == 0 ? 0 : t_report / n_report;
				vector<double> report = { static_cast<double>(n_used - n_used_since_report),
					t_data / n_used, avgtd + avgtu, avgtr, loss - loss_since_report };
				// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
				// VLOG(2) << "  w send report: " << n_used << " " << t_data / n_used << " " << avgtd
				//  		<< " " << avgtu << " " << avgtr;
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
			stat.t_dlt_send += tmr.elapseSd();
			n_probe += n_used;
		}
		// send init loss for current probe
		if (suLossReq.ready()) {
			Timer tmr;
			// VLOG(2) << " calc loss 4 cur probe batch from " << prevStart << " for " << n_probe;
			lock_guard<mutex> lk(mParam); // lock prameter
			LB = calcLoss(0, n_bench) / n_bench;
			sendLoss(LB);
			t_calcLoss += tmr.elapseSd();
			VLOG(2) << "Probe recalculate L for n_bench " << n_bench << "\ttime: " << tmr.elapseSd();
			prevStart = dataPointer;
			n_probe = 0;
			suLossReq.reset();
		}
		++iter;
	}
}

void Worker::papOnlineProbeFile()
{
	Timer tmr;
	// DVLOG(3) << "current parameter: " << model.getParameter().weights;
	double dly = getSpeedFactor();
	clearDelta();
	resumeTrain();
	Trainer::DeltaResult dr = trainer->batchDelta(allowTrain, dataPointer, localReportSize, false, dly);
	updatePointer(dr.n_scanned, dr.n_reported);
	size_t n_used = dr.n_reported;
	DVLOG_EVERY_N(ln, 2) << "  calculate delta with " << dr.n_scanned << " data points";
	if(dr.n_reported != 0){
		accumulateDelta(dr.delta);
	}
	double t_data = tmr.elapseSd();
	stat.t_dlt_calc += t_data;
	if(n_used != 0){
		tmr.restart();
		double wtd = t_data / n_used;
		vector<double> report = { static_cast<double>(n_used),
			wtd, wtd, wtd, dr.loss };
		// format: #-processed-data-points, time-per-data-point, time-per-delta-sending, time-per-report-sending, loss
		DVLOG_EVERY_N(ln, 2) << "  send report: " << report;
		sendReport(report);
		++n_report;
		t_report += tmr.elapseSd();
	}
	if(hasNewParam){
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
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
