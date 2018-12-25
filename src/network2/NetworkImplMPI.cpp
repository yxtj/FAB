/*
 * NetworkImplMPI.cpp
 *
 *  Created on: Nov 29, 2015
 *      Author: tzhou
 */
#include "NetworkImplMPI.h"

using namespace std;
//namespace mpi = boost::mpi;

static void CrashOnMPIError(MPI_Comm * c, int * errorCode, ...){
	char buffer[1024];
	int size = 1024;
	MPI_Error_string(*errorCode, buffer, &size);
	throw runtime_error("MPI function failed: " + string(buffer));
}

NetworkImplMPI::NetworkImplMPI(int argc, char* argv[]): id_(-1),size_(0){
//	if(!getenv("OMPI_COMM_WORLD_RANK") && !getenv("PMI_RANK")){
//		throw runtime_error("Not running under OpenMPI or MPICH");
//	}
	int mt_provide;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &mt_provide);
	world = MPI_COMM_WORLD;

	MPI_Errhandler handler;
	MPI_Comm_create_errhandler(&CrashOnMPIError, &handler);
//	MPI_Errhandler_create(&CrashOnMPIError, &handler);
	MPI_Comm_set_errhandler(world, handler);

	MPI_Comm_rank(world, &id_);
	MPI_Comm_size(world, &size_);
}

NetworkImplMPI* NetworkImplMPI::self = nullptr;
void NetworkImplMPI::Init(int argc, char * argv[])
{
	self = new NetworkImplMPI(argc, argv);
}

NetworkImplMPI* NetworkImplMPI::GetInstance(){
	return self;
}

void NetworkImplMPI::Shutdown(){
	int flag;
	MPI_Finalized(&flag);
	if(!flag){
		MPI_Finalize();
	}
	delete self;
	self=nullptr;
}

////
// Transmitting functions:
////

bool NetworkImplMPI::probe(TaskHeader* hdr){
	MPI_Status st;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, world, &flag, &st);
	if(!flag)
		return false;
	hdr->src_dst=st.MPI_SOURCE;
	hdr->type=st.MPI_TAG;
	MPI_Get_count(&st, MPI_BYTE, &hdr->nBytes);
	return true;
}

std::pair<std::string, TaskBase> NetworkImplMPI::receive()
{
	MPI_Status st;
	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, world, &st);
	int nBytes;
	MPI_Get_count(&st, MPI_BYTE, &nBytes);
	string data(nBytes, '\0');
	MPI_Recv(const_cast<char*>(data.data()), nBytes, MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, world, &st);
	return make_pair(move(data), TaskBase{st.MPI_SOURCE, st.MPI_TAG});
}

std::string NetworkImplMPI::receive(const TaskHeader* hdr){
	string data(hdr->nBytes,'\0');
//	world.Recv(const_cast<char*>(data.data()), hdr->nBytes, MPI_BYTE, hdr->src_dst, hdr->type);
	MPI_Status st;
	MPI_Recv(const_cast<char*>(data.data()), hdr->nBytes, MPI_BYTE, hdr->src_dst, hdr->type, world, &st);
	return data;
}
std::string NetworkImplMPI::receive(int dst, int type, const int nBytes){
	string data(nBytes,'\0');
	// address transfer
	dst=TransformSrc(dst);
	type=TransformTag(type);
//	world.Recv(const_cast<char*>(data.data()), nBytes, MPI_BYTE, dst, type);
	MPI_Status st;
	MPI_Recv(const_cast<char*>(data.data()), nBytes, MPI_BYTE, dst, type, world, &st);
	return data;
}

void NetworkImplMPI::send(const Task* t){
	MPI_Send(t->payload.data(), t->payload.size(), MPI_BYTE, t->src_dst, t->type, t->type);
}

void NetworkImplMPI::broadcast(const Task* t){
	//MPI_IBcast does not support tag
	const int& myid = id_;
	for(int i = 0; i < size(); ++i){
		if(i != myid){
			//make sure each pointer given to send() is unique
			Task* t2=new Task(*t);
			t2->src_dst=i;
			send(t2);
		}
	}
	delete t;
}
