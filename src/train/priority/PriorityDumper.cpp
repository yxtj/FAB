#include "PriorityDumper.h"

using namespace std;

bool PriorityDumper::init(const std::string& fname, bool obinary, bool ofloat,
	bool resume, std::streampos pos)
{
	if(obinary){
		if(ofloat){
			pfd = &PriorityDumper::dumpFloat_d;
			pff = &PriorityDumper::dumpFloat_f;
		} else{
			pfd = &PriorityDumper::dumpDouble_d;
			pff = &PriorityDumper::dumpDouble_f;
		}
	} else{
		pfd = &PriorityDumper::dumpCSV_d;
		pff = &PriorityDumper::dumpCSV_f;
	}
	ios_base::openmode f = ios::out;
	if(obinary)
		f |= ios::binary;
	if(resume && pos != 0){
		fout.open(fname, f | ios::in);
		fout.seekp(pos);
	} else{
		fout.open(fname, f);
	}
	return fout.is_open();
}

void PriorityDumper::dumpCSV_d(const std::vector<double>& priority){
	for(auto& v : priority){
		fout << v << ",";
	}
	fout.seekp(-1, ios::cur);
	fout << "\n";
}

void PriorityDumper::dumpCSV_f(const std::vector<float>& priority){
	for(auto& v : priority){
		fout << v << ",";
	}
	fout.seekp(-1, ios::cur);
	fout << "\n";
}

void PriorityDumper::dumpDouble_d(const std::vector<double>& priority){
	fout.write((const char*)priority.data(), priority.size() * sizeof(double));
}

void PriorityDumper::dumpDouble_f(const std::vector<float>& priority){
	for(auto& v : priority){
		double f = static_cast<double>(v);
		fout.write((const char*)&f, sizeof(double));
	}
}

void PriorityDumper::dumpFloat_d(const std::vector<double>& priority){
	for(auto& v : priority){
		float f = static_cast<float>(v);
		fout.write((const char*)&f, sizeof(float));
	}
}

void PriorityDumper::dumpFloat_f(const std::vector<float>& priority){
	fout.write((const char*)priority.data(), priority.size() * sizeof(float));
}
