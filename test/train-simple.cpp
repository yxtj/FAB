#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "data/DataHolder.h"
#include "model/Model.h"
#include "train/GD.h"

using namespace std;

void showParameter(const Parameter& m){
	for(auto& v : m.weights){
		cout << v << ", ";
	}
	cout << "\n";
}

int main(int argc, char* argv[]){
	cout << "start" << endl;
	string prefix = argc >= 2 ? string(argv[1]) : "E:/Code/FSB/dataset/";
	string name = "affairs.csv";
	DataHolder dh(1, 0);
	try{
		dh.load(prefix + name, ",", { 0 }, { 9 }, true, false);
	}catch(exception& e){
		cerr << "load error:\n" << e.what() << endl;
		return 1;
	}
	dh.normalize(false);
	
	cout << "training" << endl;
	int nx = dh.xlength();
	Model m;
	m.init("lr", to_string(nx), 0.01);
	if(!m.checkData(dh.xlength(), dh.ylength())){
		cerr << "data size does not match model" << endl;
		return 2;
	}
	cout << "parameters:\n";
	showParameter(m.getParameter());

	GD trainer;
	trainer.bindModel(&m);
	trainer.bindDataset(&dh);
	trainer.setRate(0.1);

	for(int i = 0; i < 5; i++){
		trainer.train(i*500, 500);
		cout << i << " iter: " << trainer.loss() << endl;
		cout << "parameters:\n";
		showParameter(m.getParameter());
	}

	cout << "full trainning:" << endl;
	for(int i = 0; i < 300; ++i){
		trainer.train();
		if(i % 10 == 0){
			cout << i << " iter: " << trainer.loss() << endl;
			cout << "parameters:\n";
			showParameter(m.getParameter());
		}
	}

	return 0;
}

