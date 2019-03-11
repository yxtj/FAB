#include "NetworkThread.h"
#include "NetworkImplMPI.h"
#include "logging/logging.h"
#include <chrono>
using namespace std;

double FLAGS_sleep_time = 0.001;

static inline void Sleep(){
	this_thread::sleep_for(chrono::duration<double>(FLAGS_sleep_time));
}

NetworkThread::NetworkThread() :
	stat_send_pkg(0), stat_recv_pkg(0),
	stat_send_byte(0), stat_recv_byte(0),
	stat_send_time(0.0), stat_recv_time(0.0),
	stat_time_serial(0.0),
	running(false), done(false), net(nullptr)
{
	net = NetworkImplMPI::GetInstance();
	pending_sends_ = &ps_buffer_[0];

	running = true;
	t_ = thread(&NetworkThread::Run, this);
	//pt_->detach();
}

int NetworkThread::id() const{
	return net->id();
}
int NetworkThread::size() const{
	return net->size();
}

bool NetworkThread::active() const{
	return net->unconfirmedTaskNum() > 0 ||
			ps_buffer_[0].size() > 0 || ps_buffer_[1].size() > 0;
}
size_t NetworkThread::pending_pkgs() const{
	return net->unconfirmedTaskNum()+ps_buffer_[0].size()+ps_buffer_[1].size();
}
int64_t NetworkThread::pending_bytes() const{
	int64_t t = net->unconfirmedBytes();

	lock_guard<recursive_mutex> sl(ps_lock);
	for(const vector<pair<Task*,bool>>& vec:ps_buffer_){
		for(const pair<Task*,bool>& p:vec)
			t+=p.first->payload.size();
	}

	return t;
}

//size_t NetworkThread::unpicked_pkgs() const{
//	return receive_buffer.size();
//}
int64_t NetworkThread::unpicked_bytes() const{
	int64_t t=0;
	lock_guard<recursive_mutex> rl(rec_lock);
	for(const auto& p: receive_buffer){
		t+=p.first.size();
	}
	return t;
}

void NetworkThread::Run(){
	TaskHeader hdr;
	unsigned cnt_idle_loop=0;
	static constexpr unsigned SLEEP_CNT=256;
	done=false;
	while(running){
		bool idle=true;
		//receive
		if(!pause_ && net->probe(&hdr)){
			string data = net->receive(&hdr);
			DVLOG(5)<<"RP from "<<hdr.src_dst<<" type "<<hdr.type;
			lock_guard<recursive_mutex> sl(rec_lock);

			receive_buffer.push_back(make_pair(move(data),TaskBase{hdr.src_dst, hdr.type}));
			idle=false;
		}
		//clear useless send buffer
		net->collectFinishedSend();
		//send
		/* bunch send: */
		if(!pause_ && !pending_sends_->empty()){
			//two-buffers-swapping implementation for better performance
			vector<pair<Task*, bool>>* pv;
			{
				lock_guard<recursive_mutex> sl(ps_lock);
				pv = pending_sends_;
				pending_sends_ = &ps_buffer_[++ps_idx_ % 2];
			}
			auto end_it=pv->end();
			for(auto it = pv->begin(); it!=end_it; ++it){
				DVLOG(5)<<"SP to "<< it->first->src_dst<<" type "<< it->first->type;
				if(it->second) // broadcast
					net->broadcast(it->first);
				else
					net->send(it->first);
			}
			pv->clear();
		}else{
			if(idle && ++cnt_idle_loop%SLEEP_CNT==0)
				Sleep();
		}
	}
	done=true;
}

bool NetworkThread::checkReceiveQueue(std::string& data, TaskBase& info){
	if(!receive_buffer.empty()){
		lock_guard<recursive_mutex> sl(rec_lock);
		if(receive_buffer.empty())
			return false;

		tie(data,info)=receive_buffer.front();
		receive_buffer.pop_front();
		return true;
	}
	return false;
}

void NetworkThread::readAny(string& data, int *srcRet, int *typeRet){
	while(!tryReadAny(data, srcRet, typeRet)){
		Timer tmr;
		Sleep();
		stat_recv_time += tmr.elapseSd();
	}
}
bool NetworkThread::tryReadAny(string& data, int *srcRet, int *typeRet){
	Timer tmr;
	bool found = false;
	TaskBase info;
	if(checkReceiveQueue(data,info)){
		if(srcRet) *srcRet = info.src_dst;
		if(typeRet) *typeRet = info.type;
		++stat_recv_pkg;
		stat_recv_byte += data.size();
		found = true;
	}
	stat_recv_time += tmr.elapseSd();
	return found;
}

// Enqueue the given request to pending buffer for transmission.
void NetworkThread::send(Task *req){
	int size = req->payload.size();
	lock_guard<recursive_mutex> sl(ps_lock);
	pending_sends_->emplace_back(req, false);
	++stat_send_pkg;
	stat_send_byte += size;
}

void NetworkThread::broadcast(Task* req) {
	int size = req->payload.size();
	lock_guard<recursive_mutex> sl(ps_lock);
	pending_sends_->emplace_back(req, true);
	//net->broadcast(req);
	stat_send_pkg += this->size() - 1;
	stat_send_byte += size * (this->size() - 1);
}

void NetworkThread::flush() {
	while(active()) {
		Sleep();
	}
}

// ---------------- singleton related ------------------

NetworkThread* NetworkThread::self = nullptr;

NetworkThread* NetworkThread::GetInstance(){
	return self;
}

void NetworkThread::Init(int argc, char* argv[]){
	NetworkImplMPI::Init(argc, argv);
	self = new NetworkThread();
	atexit(&NetworkThread::Terminate);
}

void NetworkThread::Shutdown() {
	if(self != nullptr) {
		NetworkThread* p = nullptr;
		swap(self, p); // use the swap primitive to preform safe deletion
		if(p->running) {
			p->flush();	//finish all the sending
			p->running = false;
			//wait for Run() to exit
			while(!p->done) {
				Sleep();
			}
			p->t_.join();
			p->net = nullptr;
			NetworkImplMPI::Shutdown();
		}
	}
}

void NetworkThread::Terminate()
{
	if(self != nullptr) {
		NetworkThread* p = nullptr;
		swap(p, self);
		p->running = false;
		p->t_.join();
		p->net = nullptr;
		delete p;
		NetworkImplMPI::Shutdown();
	}
}
