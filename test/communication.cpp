#include <iostream>
#include <string>
#include <mpi.h>
#include "logging/logging.h"
#include "network/NetworkThread.h"
#include "network/RPCInfo.h"
#include "message/MType.h"

using namespace std;

void assert(bool flag, const string msg){
	LOG_IF(!flag, ERROR) << msg;
}

void sendReply(NetworkThread* net, RPCInfo& info){
	LOG(INFO) << "  send reply to type " << info.tag;
	net->send(info.source, MType::CReply, info.tag);
}

void checkReply(NetworkThread* net, int type){
	string data;
	int src;
	int tag;
	net->readAny(data, &src, &tag);
	int rtag = deserialize<int>(data);
	LOG(INFO) << "receive src=" << src << " tag=" << tag << " reply=" << rtag;
	assert(tag == MType::CReply, "  (checkReply): message is not a reply. It is " + to_string(tag));
	assert(rtag == type, "  (checkReply): reply type does not match. "
		"want: " + to_string(type) + " get: " + to_string(rtag));
}

void basicMpiTest(){
	MPI_Comm world = MPI_COMM_WORLD;
	int id;
	MPI_Comm_rank(world, &id);
	string data(100, '\0');
	MPI_Status st;
	setLogThreadName("T" + to_string(id));
	if(id == 0){
		data = "zhang-san";
		LOG(INFO) << "send " << data << " via 12";
		MPI_Send(data.c_str(), static_cast<int>(data.size()), MPI_BYTE, 1, 12, world);
		//checkReply(net, 12);
		//net->readAny(data, &info.source, &info.tag);
		//LOG(INFO) << "receive src=" << info.source << " tag=" << info.tag << " data=" << deserialize<int>(data);
	} else{
		LOG(INFO) << "waiting for message";
		MPI_Recv(const_cast<char*>(data.data()), 100, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, world, &st);
		LOG(INFO) << "receive src=" << st.MPI_SOURCE << " tag=" << st.MPI_TAG << " data=" << data;
		//sendReply(net, info);
		//net->send(info.source, MType::CReply, info.tag);
	}
}

void networkModuleTest(){
	NetworkThread* net = NetworkThread::GetInstance();
	string data;
	RPCInfo info;
	if(net->id() == 0){
		info.dest = 0;
		setLogThreadName("T0");
		LOG(INFO) << "send zhang-san via 12";
		net->send(1, 12, "zhang-san");
		checkReply(net, 12);
		//net->readAny(data, &info.source, &info.tag);
		//LOG(INFO) << "receive src=" << info.source << " tag=" << info.tag << " data=" << deserialize<int>(data);
	} else{
		info.dest = 1;
		setLogThreadName("T1");
		net->readAny(data, &info.source, &info.tag);
		LOG(INFO) << "receive src=" << info.source << " tag=" << info.tag << " data=" << data;
		sendReply(net, info);
		//net->send(info.source, MType::CReply, info.tag);
	}
}

int main(int argc, char* argv[]){
	initLogger(argc, argv);
	NetworkThread::Init(argc, argv);
	NetworkThread* net = NetworkThread::GetInstance();
	LOG(INFO) << "size=" << net->size() << " id=" << net->id();
	if(net->size() != 2){
		cerr << "2 MPI instance should be started";
		return 2;
	}
	LOG(INFO) << "basic MPI test";
	basicMpiTest();
	LOG(INFO) << "network module test";
	networkModuleTest();
	NetworkThread::Shutdown();
	return 0;
}
