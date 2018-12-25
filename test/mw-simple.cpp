#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "data/DataHolder.h"
#include "model/Model.h"
#include "train/GD.h"

using namespace std;

class DummyNetwork{
	Parameter buffer;
	vector<double> grad;

public:
	void send_p(const Parameter& org){
		buffer = org;
	}
	void receive_p(Parameter& res){
		res = buffer;
	}
	void send_g(const vector<double>& org){
		grad = org;
	}
	void receive_g(vector<double>& res){
		res = grad;
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

void test_1m1w(DataHolder& dh){
	cout << "simulating" << endl;
	DummyNetwork net;

	// PM has: param;
	Parameter m_param;
	m_param.init(dh.xlength(), 0.05, 0.1, 0);
	Model m_model;
	showParameter("  m", m_param);
	m_model.setParameter(m_param);
	vector<double> m_grad;

	// Worker has: trainer. fetches and updates
	Model w_model;
	Parameter w_param;
	showParameter("  w", w_param);
	w_model.setParameter(w_param);
	GD trainer;
	trainer.bindModel(&w_model);
	trainer.bindDataset(&dh);
	trainer.setRate(0.1);
	vector<double> w_grad;

	// start working
	cout << "start working" << endl;
	cout << "  init parameters" << endl;
	net.send_p(m_param);
	net.receive_p(w_param);
	w_model.setParameter(w_param);
	showParameter("  w p:", w_model.getParameter());
	cout << "  loss 0: " << w_model.loss(dh.get(0)) << endl;

	cout << "train 1" << endl;
	w_grad = trainer.batchDelta(500, 500, true);
	w_model.accumulateParameter(w_grad);
	cout << "  loss 1 w: " << w_model.loss(dh.get(0)) << endl;
	cout << "  cooridinating" << endl;
	showParameter("  w p:", w_model.getParameter());
	net.send_g(w_grad);
	showParameter("  m p before:", m_model.getParameter());
	net.receive_g(m_grad);
	m_model.accumulateParameter(m_grad);
	showParameter("  m p after:", m_model.getParameter());
	cout << "  loss 1 m: " << m_model.loss(dh.get(0)) << endl;

	cout << "train 2" << endl;
	w_grad = trainer.batchDelta(1000, 500, true);
	w_model.accumulateParameter(w_grad);
	cout << "  loss 2 w: " << w_model.loss(dh.get(0)) << endl;
	cout << "  cooridinating" << endl;
	showParameter("  w p:", w_model.getParameter());
	net.send_g(w_grad);
	showParameter("  m p before:", m_model.getParameter());
	net.receive_g(m_grad);
	m_model.accumulateParameter(m_grad);
	showParameter("  m p after:", m_model.getParameter());
	cout << "  loss 2 m: " << m_model.loss(dh.get(0)) << endl;
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
};

void test_1m2w(DataHolder& dh){
	const size_t nw = 2; // number of workers
	const size_t bs = 500; // batch size
	size_t nx = dh.xlength();
	DummyNetwork net;
	Runner m(dh), w1(dh), w2(dh);

	// init of master
	m.bf_param.init(nx, 0.05, 0.1, 0);
	m.model.setParameter(m.bf_param);
	net.send_p(m.model.getParameter());
	// init of workers
	net.receive_p(w1.bf_param);
	w1.model.setParameter(w1.bf_param);
	net.receive_p(w2.bf_param);
	w2.model.setParameter(w2.bf_param);
	cout << "Initialize, loss: " << m.trainer.loss() << endl;
	showParameter("  m p", m.model.getParameter());
	showParameter("  w1 p", w1.model.getParameter());
	showParameter("  w2 p", w2.model.getParameter());

	// iteration 1
	size_t iter = 1;
	cout << "Train " << iter << endl;
	w1.bf_grad = w1.trainer.batchDelta((iter - 1)*nw*bs, bs, true);
	w1.trainer.applyDelta(w1.bf_grad);
	cout << "  w1 loss: " << w1.trainer.loss() << endl;
	showParameter("  w1 p", w1.model.getParameter());
	w2.bf_grad = w2.trainer.batchDelta((iter - 1)*nw*bs + bs, bs, true);
	w2.trainer.applyDelta(w2.bf_grad);
	cout << "  w2 loss: " << w2.trainer.loss() << endl;
	showParameter("  w2 p", w2.model.getParameter());

	cout << "  coordinating - gather delta (w->m)" << endl;
	net.send_g(w1.bf_grad);
	net.receive_g(m.bf_grad);
	m.model.accumulateParameter(m.bf_grad, 1.0 / nw);
	net.send_g(w2.bf_grad);
	net.receive_g(m.bf_grad);
	m.model.accumulateParameter(m.bf_grad, 1.0 / nw);
	cout << "  m loss: " << m.trainer.loss() << endl;
	showParameter("  m p", m.model.getParameter());

	cout << "  coordinating - broadcast result (m->w)" << endl;
	net.send_p(m.model.getParameter());
	net.receive_p(w1.bf_param);
	w1.model.setParameter(w1.bf_param);
	cout << "  w1 loss: " << w1.trainer.loss() << endl;
	showParameter("  w1 p", w1.model.getParameter());
	net.receive_p(w2.bf_param);
	w2.model.setParameter(w2.bf_param);
	cout << "  w2 loss: " << w2.trainer.loss() << endl;
	showParameter("  w2 p", w2.model.getParameter());

	// iteration 2
	++iter;
	cout << "Train " << iter << endl;
	w1.bf_grad = w1.trainer.batchDelta((iter - 1)*nw*bs, bs, true);
	w1.trainer.applyDelta(w1.bf_grad);
	cout << "  w1 loss: " << w1.trainer.loss() << endl;
	showParameter("  w1 p", w1.model.getParameter());
	w2.bf_grad = w2.trainer.batchDelta((iter - 1)*nw*bs + bs, bs, true);
	w2.trainer.applyDelta(w2.bf_grad);
	cout << "  w2 loss: " << w2.trainer.loss() << endl;
	showParameter("  w2 p", w2.model.getParameter());

	cout << "  coordinating - gather delta (w->m)" << endl;
	net.send_g(w1.bf_grad);
	net.receive_g(m.bf_grad);
	m.model.accumulateParameter(m.bf_grad, 1.0 / nw);
	net.send_g(w2.bf_grad);
	net.receive_g(m.bf_grad);
	m.model.accumulateParameter(m.bf_grad, 1.0 / nw);
	cout << "  m loss: " << m.trainer.loss() << endl;
	showParameter("  m p", m.model.getParameter());

	cout << "  coordinating - broadcast result (m->w)" << endl;
	net.send_p(m.model.getParameter());
	net.receive_p(w1.bf_param);
	w1.model.setParameter(w1.bf_param);
	cout << "  w1 loss: " << w1.trainer.loss() << endl;
	showParameter("  w1 p", w1.model.getParameter());
	net.receive_p(w2.bf_param);
	w2.model.setParameter(w2.bf_param);
	cout << "  w2 loss: " << w2.trainer.loss() << endl;
	showParameter("  w2 p", w2.model.getParameter());
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
	
	cout << "Test 1, 1 master 1 worker." << endl;
	test_1m1w(dh);

	test_1m2w(dh);

	return 0;
}

