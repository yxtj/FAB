#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include "data/DataHolder.h"
#include "model/Model.h"
#include "train/GD.h"

using namespace std;

class DummyNetworkT {
	Parameter param;
	vector<double> grad;

	struct Tunnel{
		bool has_data;
		mutex m_r, m_w;
		condition_variable cv_r, cv_w;
	};
	Tunnel tp;
	Tunnel tg;
public:
	void send_p(const Parameter& org){
		unique_lock<mutex> lk(tp.m_w);
		tp.cv_w.wait(lk, [&]{return !tp.has_data; });
		param = org;
		tp.has_data = true;
		tp.cv_r.notify_one();
	}
	void receive_p(Parameter& res){
		unique_lock<mutex> lk(tp.m_r);
		tp.cv_r.wait(lk, [&]{return tp.has_data; });
		res = param;
		tp.has_data = false;
		tp.cv_w.notify_one();
	}
	void send_g(const vector<double>& org){
		unique_lock<mutex> lk(tg.m_w);
		tg.cv_w.wait(lk, [&]{return !tg.has_data; });
		grad = org;
		tg.has_data = true;
		tg.cv_r.notify_one();
	}
	void receive_g(vector<double>& res){
		unique_lock<mutex> lk(tg.m_r);
		tg.cv_r.wait(lk, [&]{return tg.has_data; });
		res = grad;
		tg.has_data = false;
		tg.cv_w.notify_one();
	}
};

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

struct Runner{
	Model model;
	GD trainer;
	// buffer data
	Parameter bf_param;
	vector<double> bf_grad;
public:
	Runner(const DataHolder& dh) {
		trainer.bindModel(&model);
		trainer.bindDataset(&dh);
	}
	vector<double> getDalta(const size_t start, const size_t cnt){
		atomic_bool flag;
		bf_grad = trainer.batchDelta(flag, start, cnt, true).delta;
		return bf_grad;
	}
	void applyDelta(const double factor){
		model.accumulateParameter(bf_grad, factor);
	}
};

DummyNetworkT net;

void master_thread(const DataHolder& dh, const size_t nw){
	Runner m(dh);
	size_t nx = dh.xlength();
	cout << "master started" << endl;
	m.bf_param.init(nx, 0.05, 0.1, 0);
	m.model.setParameter(m.bf_param);

	cout << "Initialize, loss: " << m.trainer.loss() << endl;
	showParameter("  m p ", m.model.getParameter());
	for(size_t i = 0; i < nw; ++i){
		net.send_p(m.model.getParameter());
	}

	cout << "Train 1 - m" << endl;
	for(size_t i = 0; i < nw; ++i){
		net.receive_g(m.bf_grad);
		cout << "  m receive g (" << i+1 << "/" << nw << ")" << endl;
		m.applyDelta(1.0 / nw);
	}
	cout << "  m loss: " << m.trainer.loss() << endl;
	showParameter("  m p ", m.model.getParameter());
	cout << "  m send p" << endl;
	for(size_t i = 0; i < nw; ++i){
		net.send_p(m.model.getParameter());
	}
	
}

void worker_thread(const DataHolder& dh, const size_t bs, const size_t nw, const size_t id){
	Runner w(dh);
	string wname = "w-" + to_string(id);
	cout << "worker " << id << " started" << endl;
	net.receive_p(w.bf_param);
	w.model.setParameter(w.bf_param);
	showParameter("  "+wname+" p received ", w.model.getParameter());

	size_t iter = 1;
	cout << "Train 1 - " << id << endl;
	w.getDalta((iter - 1)*nw*bs + id * bs, bs);
	w.applyDelta(1.0);
	cout << "  " << wname << " loss: " << w.trainer.loss() << endl;
	showParameter("  " + wname + " p ", w.model.getParameter());

	cout << "  " << wname << " send g" << endl;
	net.send_g(w.bf_grad);
	cout << "  " << wname << " wait p" << endl;
	net.receive_p(w.bf_param);
	cout << "  " << wname << " receive p" << endl;
	w.model.setParameter(w.bf_param);
	cout << "  " << wname << " loss: " << w.trainer.loss() << endl;
	showParameter("  " + wname + " p ", w.model.getParameter());

}

void test_1m2w(DataHolder& dh){
	const size_t nw = 2; // number of workers
	const size_t bs = 500; // batch size
	
	thread tm(master_thread, dh, nw);
	thread tw1(worker_thread, dh, bs, nw, 0);
	thread tw2(worker_thread, dh, bs, nw, 1);

	tm.join();
	tw1.join();
	tw2.join();
}

int main(int argc, char* argv[]){
	cout << "start" << endl;
	string prefix = argc > 1 ? argv[1] : "E:/Code/FSB/dataset/";
	string name = argc > 2 ? argv[2] : "affairs.csv";
	DataHolder dh(1, 0);
	try{
		dh.load(prefix + name, ",", { 0 }, { 9 }, true, false);
	}catch(exception& e){
		cerr << "load error:\n" << e.what() << endl;
		return 1;
	}
	dh.normalize(false);
	
	cout << "Test 1, 1 master 2 worker." << endl;
	test_1m2w(dh);

	return 0;
}

