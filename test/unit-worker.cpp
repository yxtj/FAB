#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <mpi.h>
//#include <cassert>
#include "logging/logging.h"
#include "common/ConfData.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "data/DataHolder.h"
#include "distr/Worker.h"

using namespace std;

void assert(bool flag, const string msg){
	LOG_IF(!flag, ERROR) << msg;
}

void sendReply(NetworkThread* net, RPCInfo& info){
	LOG(INFO)<< "  send reply to type " << info.tag;
	net->send(info.source, MType::CReply, info.tag);
}

void checkReply(NetworkThread* net, int type){
	string data;
	int src;
	int tag;
	net->readAny(data, &src, &tag);
	int rtag = deserialize<int>(data);
	LOG(INFO) << "receive src=" << src << " tag=" << tag << " reply=" << rtag;
	assert(tag == MType::CReply, "  (checkReply): message is not a reply. It is "+to_string(tag));
	assert(rtag == type, "  (checkReply): reply type does not match. "
		"want: "+to_string(type)+" get: "+to_string(rtag));
}

void showParameter(const string& prefix, const Parameter& m){
	double s1 = 0.0, s2 = 0.0;
	for(auto&v : m.weights){
		s1 += v >= 0 ? v : -v;
		s2 += v * v;
	}
	s2 = sqrt(s2);
	if(!prefix.empty())
		cout << prefix << s1 << " , " << s2 << "\n";
	for(auto& v : m.weights){
		cout << v << ", ";
	}
	cout << "\n";
}

void master_thread(){
	NetworkThread* net = NetworkThread::GetInstance();
	string data;
	RPCInfo info;
	info.dest = 0;
	IDMapper wm;
	size_t nx = 0;
	int lid = 0;
	double factor = 1.0;

	// online message
	LOG(INFO) << "Wait online message";
	net->readAny(data, &info.source, &info.tag);
	LOG(INFO) << "  receive from " << info.source << " to " << info.dest << " type " << info.tag << ". content: " << deserialize<int>(data);
	assert(info.tag == MType::COnline, "does not receive online message");
	assert(lid == deserialize<int>(data), "id not match");
	wm.registerID(info.source, lid);
	int nid = wm.lid2nid(lid);
	LOG(INFO) << "  Send reply: online message";
	sendReply(net, info);

	// send worker list
	LOG(INFO) << "Send worker list";
	net->send(nid, MType::CWorkers, wm.list());
	checkReply(net, MType::CWorkers);

	// get xlength
	LOG(INFO)<<"gather xlength";
	net->readAny(data, &info.source, &info.tag);
	nx = deserialize<size_t>(data);
	LOG(INFO)<<"got xlength="<<nx;
	sendReply(net, info);

	Model m;
	m.init("lr", to_string(nx), 0.01);
	if(!m.checkData(nx, 1))
		LOG(FATAL) << "data size does not match model";
	vector<double> delta;

	// send parameter
	LOG(INFO) << "  Send parameter 0";
	net->send(nid, MType::DParameter, m.getParameter().weights);
	checkReply(net, MType::DParameter);

	int iter = 1;
	// train 1
	while(iter<=2){
		LOG(INFO)<<"Iteration: "<<iter<<" waiting for delta message ";
		net->readAny(data, &info.source, &info.tag);
		assert(info.tag==MType::DDelta, "receive type is not delta message: "+to_string(info.tag));
		delta = deserialize<vector<double>>(data);
		m.accumulateParameter(delta, factor);
		showParameter("train "+to_string(iter)+". ", m.getParameter());
		LOG(INFO)<<"  Send reply: delta";
		sendReply(net, info);
		LOG(INFO)<<"  Send parameter "<<iter;
		net->send(nid, MType::DParameter, m.getParameter().weights);
		checkReply(net, MType::DParameter);
		++iter;
	}

	// termination
	LOG(INFO) << "Send termination signal";
	net->send(nid, MType::CTerminate, "");
	LOG(INFO)<<"Taking the possible final delta";
	net->readAny(data, &info.source, &info.tag);
	if(info.tag==MType::CReply){
		LOG(INFO)<<"  received a reply";
		checkReply(net, MType::CTerminate);
	} else if(info.tag == MType::DDelta){
		LOG(INFO)<<"  received a delta";
		delta = deserialize<vector<double>>(data);
		m.accumulateParameter(delta, factor);
		LOG(INFO)<<"  waiting reply of CTermination signal";
		checkReply(net, MType::CTerminate);
	}
	LOG(INFO)<<"Waiting worker closed notification";
	net->readAny(data, &info.source, &info.tag);
	assert(info.tag == MType::CClosed, "last message is not CClosed: "+to_string(info.tag));
}

void worker_thread(ConfData& conf, DataHolder& dh){
	Worker w;
	w.init(&conf, 0);
	w.bindDataset(&dh);

	cout << "Starting worker" << endl;
	w.run();
}

void test_worker(DataHolder& dh, const int nid){
	ConfData conf;
	conf.batchSize = 500;
	conf.nw = 1;
	conf.optimizer = "gd";
	conf.optimizerParam = { "1" };
	conf.mode = "sync";
	conf.adjustSpeedHetero = conf.adjustSpeedRandom = false;

	if(nid == 0){
		setLogThreadName("M");
		master_thread();
	} else{
		setLogThreadName("W0");
		worker_thread(conf, dh);
	}
	
}

int main(int argc, char* argv[]){
	initLogger(argc, argv);
	NetworkThread::Init(argc, argv);
	NetworkThread* net = NetworkThread::GetInstance();
	cout << "size=" << net->size() << " id=" << net->id() << endl;
	if(net->size() != 2){
		cerr << "2 MPI instance should be started" << endl;
		return 2;
	}
#ifndef NDEBUG
	if(net->id() == 0){
		DLOG(DEBUG) << "pause.";
		DLOG(DEBUG) << cin.get();
	}
#endif

	cout << "load data" << endl;
	string prefix = argc > 1 ? argv[1] : "E:/Code/FSB/dataset/";
	//string name = argc > 2 ? argv[2] : "affairs.csv";
	string name = "affairs.csv";
	DataHolder dh(1, 0);
	try{
		if(net->id() == 1)
			dh.load(prefix + name, ",", { 0 }, { 9 }, true, false);
	} catch(exception& e){
		cerr << "load error:\n" << e.what() << endl;
		return 1;
	}
	dh.normalize(false);

	cout << "Test" << endl;
	test_worker(dh, net->id());

	NetworkThread::Shutdown();
	//NetworkThread::Terminate();
	return 0;
}
