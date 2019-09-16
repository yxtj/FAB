#include <iostream>
#include <string>
#include <fstream>
#include "data/DataHolder.h"

using namespace std;

void showLine(const DataPoint& d){
	for(const auto& line : d.x)
		for(const auto& v : line)
			cout << v << ", ";
	for(const auto& v : d.y)
		cout << v << ", ";
	cout << endl;
}

int main(int argc, char* argv[]){
	cout<<"start"<<endl;
	string prefix = argc > 1 ? argv[1] : "E:/Code/FSB/dataset/";
	string name = argc > 2 ? argv[2] : "affairs.csv";
	bool header = argc > 3 ? argv[3] == "1" : true;
	DataHolder dh(1, 0);
	try{
		dh.load(prefix + name, ",", { 0 }, { 9 }, header, true);
	}catch(exception& e){
		cerr << "load error:\n" << e.what() << endl;
		return 1;
	}

	cout << dh.size() << endl;
	showLine(dh.get(0));
	showLine(dh.get(1));
	showLine(dh.get(2));
	showLine(dh.get(6));
	showLine(dh.get(dh.size()-1));

	cout << "normalize" << endl;
	dh.normalize(false);
	showLine(dh.get(0));
	showLine(dh.get(1));
	showLine(dh.get(2));
	showLine(dh.get(6));
	showLine(dh.get(dh.size() - 1));

	return 0;
}

