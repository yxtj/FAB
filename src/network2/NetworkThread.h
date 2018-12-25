#pragma once
#include <thread>
#include <mutex>
#include <functional>
#include <string>
#include <deque>
#include <vector>
#include "Task.h"
#include "serial/serialization.h"


class NetworkImplMPI;

// Hackery to get around mpi's unhappiness with threads.  This thread
// simply polls MPI continuously for any kind of update and adds it to
// a local queue.
class NetworkThread{
public:
	// Blocking read for the given source and message type.
	void readAny(std::string& data, int *sourcsrcRete=nullptr, int *typeRet=nullptr);
	// Unblocked read for the given source and message type.
	bool tryReadAny(std::string& data, int *sosrcReturce=nullptr, int *typeRet=nullptr);

	// Enqueue the given request to pending buffer for transmission.
	template <class T>
	int send(int dst, int tag, const T& msg) {
		std::string s = serialize(msg);
		return send(new Task(dst, tag, move(s)));
	}
	int send(int dst, int tag, std::string&& msg) {
		return send(new Task(dst, tag, move(msg)));
	}

	template <class T>
	int broadcast(int tag, const T& msg) {
		std::string s = serialize(msg);
		return broadcast(new Task(Task::ANY_DST, tag, move(s)));
	}

	int id() const;
	int size() const;

	static NetworkThread *GetInstance();
	static void Init(int argc, char* argv[]);
	// finish current tasks and terminate
	static void Shutdown();
	// abandon ongoing tasks and terminate all network related functions
	static void Terminate();

	uint64_t stat_send_pkg, stat_recv_pkg;
	uint64_t stat_send_byte, stat_recv_byte;

private:
	NetworkImplMPI* net;
	// Enqueue the given request to pending buffer for transmission.
	int send(Task *req);
	// Directly (Physically) send the request.
	int sendDirect(Task *req);
	int broadcast(Task *req);

	static NetworkThread* self;
	NetworkThread();
};

