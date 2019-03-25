#include "Runner.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include <sstream>
using namespace std;

Runner::Runner()
	: msg_do_pop(true), msg_do_push(true)
{
	net = NetworkThread::GetInstance();
}

void Runner::sleep() {
	static auto d = chrono::duration<double>(0.001);
	this_thread::sleep_for(d);
}

void Runner::sleep(double seconds) {
	static auto d = chrono::duration<double>(seconds);
	this_thread::sleep_for(d);
}

void Runner::startMsgLoop(const std::string& name){
	running = true;
//	tmsg = thread(bind(&Runner::msgLoop, this, name));
	tmsg = thread(&Runner::msgLoop, this, name);
}
void Runner::stopMsgLoop(){
	running = false;
	tmsg.join();
}

void Runner::msgLoop(const std::string& name) {
	DLOG(INFO) << "Message loop started on " << name;
	if(!name.empty()){
		setLogThreadName(name);
	}
	string data;
	RPCInfo info;
	info.dest = net->id();
	while(running){
		int n = 16; // prevent spending too much time in pushing but never popping 
		while(msg_do_push && n-- >= 0 && net->tryReadAny(data, &info.source, &info.tag)){
			DVLOG(4) << "Get " << info.source << " -> " << info.dest << " (" << info.tag
				<< "), queue: " << driver.queSize() << ", net: " << net->unpicked_pkgs();
			driver.pushData(data, info);
		}
		while(msg_do_pop && !driver.empty()){
#ifndef NDEBUG
			auto& info = driver.front().second;
			DVLOG(4) << "Pop " << info.source << " -> " << info.dest << " (" << info.tag
				<< (info.tag == 0 ? "-" + to_string(deserialize<int>(driver.front().first)) : "")
				<< "), queue: " << driver.queSize() << ", net: " << net->unpicked_pkgs();
#endif
			driver.popData();
		}
		sleep();
	}
	DLOG(INFO) << "Message loop exited on " << name;
}

void Runner::finishStat()
{
	stat.n_net_send = net->stat_send_pkg;
	stat.b_net_send = net->stat_send_byte;
	stat.t_net_send = net->stat_send_time;
	stat.t_data_serial = net->stat_time_serial;

	stat.n_net_recv = net->stat_recv_pkg;
	stat.b_net_recv = net->stat_recv_byte;
	stat.t_net_recv= net->stat_recv_time;
	// stat.t_data_deserial is set in each handler function

	stat.n_iter = iter;
	stat.t_smy_work = stat.t_dlt_calc + stat.t_par_calc;
	stat.t_smy_wait = stat.t_dlt_wait + stat.t_par_wait;
}

void Runner::showStat() const
{
	ostringstream oss;
	string head = "[Stat-" + logName + "] ";
	oss << "Statistics:\n"
		<< head << "num-net-send: " << stat.n_net_send << "\tbyte-net-send: " << stat.b_net_send
		<< "\ttime-net-send: " << stat.t_net_send << "\ttime-data-serialize: " << stat.t_data_serial
		<< "\n"
		<< head << "num-net-recv: " << stat.n_net_recv << "\tbyte-net-recv: " << stat.b_net_recv
		<< "\ttime-net-recv: " << stat.t_net_recv << "\ttime-data-deserialize: " << stat.t_data_deserial
		<< "\n"
		<< head << "num-dlt-send: " << stat.n_dlt_send << "\tnum-dlt-recv: " << stat.n_dlt_recv
		<< "\ttime-dlt-calc: " << stat.t_dlt_calc << "\ttime-dlt-wait: " << stat.t_dlt_wait
		<< "\n"
		<< head << "num-par-send: " << stat.n_par_send << "\tnum-par-recv: " << stat.n_par_recv
		<< "\ttime-par-calc: " << stat.t_par_calc << "\ttime-par-wait: " << stat.t_par_wait
		<< "\n"
		<< head << "num-iteration: " << stat.n_iter << "\tnum-data-point: " << stat.n_point
		<< "\ttime-total-work: " << stat.t_smy_work << "\ttime-total-wait: " << stat.t_smy_wait
		<< "\n";
	if(logName.find("M") != logName.npos){ // master
		oss << head << "num-archive: " << stat.n_archive << "\ttime-archive: " << stat.t_archive
			<< "\ttime-per-1k-archive: " << stat.t_archive / stat.n_archive * 1000
			<< "\n"
			<< head << "time-per-1k-delta(c): " << stat.t_smy_work / stat.n_dlt_recv * 1000
			<< "\ttime-per-1k-delta(d): " << stat.t_data_deserial / stat.n_dlt_recv * 1000
			<< "\ttime-per-1k-delta: " << (stat.t_smy_work + stat.t_data_deserial) / stat.n_dlt_recv * 1000;
	} else{ // worker
		oss << head << "time-per-1k-point(c): " << stat.t_dlt_calc / stat.n_point * 1000
			<< "\ttime-per-1k-delta(s): " << stat.t_data_serial / stat.n_dlt_send * 1000
			<< "\tpoint-per-delta: " << (stat.t_data_serial / stat.n_dlt_send) / (stat.t_dlt_calc / stat.n_point);
	}	
	LOG(INFO) << oss.str();
}

void Runner::msgPausePush(){
	msg_do_push = false;
}
void Runner::msgPausePop(){
	msg_do_pop = false;
}
void Runner::msgResumePush(){
	msg_do_push = true;
}
void Runner::msgResumePop(){
	msg_do_push = true;
}

// register helpers
void Runner::regDSPImmediate(const int type, callback_t fp) {
	//driver.registerImmediateHandler(type, bind(fp, this, _1, _2));
	driver.registerImmediateHandler(type, fp);
}
void Runner::regDSPProcess(const int type, callback_t fp) {
	//driver.registerProcessHandler(type, bind(fp, this, _1, _2));
	driver.registerProcessHandler(type, fp);
}
void Runner::regDSPDefault(callback_t fp) {
	//driver.registerDefaultOutHandler(bind(fp, this, _1, _2));
	driver.registerDefaultOutHandler(fp);
}
void Runner::addRPHEach(
	const int type, std::function<void()> fun, const int n, const bool newThread)
{
	rph.addType(type,
		ReplyHandler::condFactory(ReplyHandler::EACH_ONE, n),
		fun, newThread);
	rph.activateType(type);
}
void Runner::addRPHEachSU(const int type, SyncUnit& su){
	addRPHEach(type, bind(&SyncUnit::notify, &su), static_cast<int>(nWorker), false);
}

void Runner::addRPHAny(
	const int type, std::function<void()> fun, const bool newThread)
{
	rph.addType(type,
		ReplyHandler::condFactory(ReplyHandler::ANY_ONE),
		fun, newThread);
	rph.activateType(type);
}
void Runner::addRPHAnySU(const int type, SyncUnit& su)
{
	addRPHAny(type, bind(&SyncUnit::notify, &su), false);
}

void Runner::addRPHN(
	const int type, std::function<void()> fun, const int n, const bool newThread)
{
	rph.addType(type,
		ReplyHandler::condFactory(ReplyHandler::N_TIMES, n),
		fun, newThread);
	rph.activateType(type);
}
void Runner::addRPHNSU(const int type, SyncUnit& su)
{
	addRPHN(type, bind(&SyncUnit::notify, &su), static_cast<int>(nWorker), false);
}

void Runner::sendReply(const RPCInfo& info){
	net->send(info.source, MType::CReply, info.tag);
}
