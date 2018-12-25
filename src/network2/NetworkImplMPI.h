/*
 * NetworkKernel.h
 *
 *  Created on: Nov 29, 2015
 *      Author: tzhou
 */

#pragma once

#include <deque>
#include <string>
#include <vector>
#include <mutex>
#include "Task.h"
#include <mpi.h>
//#include <boost/mpi.hpp>

/*
 * Code related with underlying implementation (MPI used here)
 */
class NetworkImplMPI{
public:
	int id() const;
	int size() const;

	// Probe is there any message, return whether find one, if any the message info in hdr.
	bool probe(TaskHeader* hdr);
	std::pair<std::string, TaskBase> receive();
	// Receive a message with given header.
	std::string receive(const TaskHeader* hdr);
	std::string receive(int dst, int type, const int nBytes);
	// Try send out a message with given hdr and content.
	void send(const Task* t);
	void broadcast(const Task* t);

	static void Init(int argc, char* argv[]);
	static NetworkImplMPI* GetInstance();
	static void Shutdown();
	static int TransformSrc(int s_d){
		//boost::mpi::any_source;
		return s_d==TaskBase::ANY_SRC ? MPI_ANY_SOURCE : s_d;
	}
	static int TransformTag(int tag){
		return tag==TaskBase::ANY_TYPE ? MPI_ANY_TAG : tag;
	}

private:
	NetworkImplMPI(int argc, char* argv[]);
	static NetworkImplMPI* self;
private:
	MPI_Comm world;
	int id_;
	int size_;
};

inline int NetworkImplMPI::id() const{
	return id_;
}
inline int NetworkImplMPI::size() const{
	return size_;
}
