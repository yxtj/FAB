#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <future>
#include "data/DataLoader.h"
#include "util/Util.h"
#include "model/Model.h"
#include "model/ParamArchiver.h"
#include "func.h"
#include "ParameterIO.h"
#include "CLI11.hpp"

using namespace std;

struct Option {
	string alg;
	string algParam;
	string fnRecord;
	bool binary = false;
	vector<int> rlines;

	string fnData;
	string dataset = "csv";
	bool dataTrainPart = false;
	vector<int> idSkip;
	vector<int> idY;
	bool normalize = false;

	string pmethod = "project";
	string fnOutput;
	bool outputBinary = false;
	bool outputFloat = true;

	bool saveMemory = false;
	int nthread = 1;
	bool resume = false;
	size_t memory = 4*(static_cast<size_t>(1)<<30); // 4-GB
	size_t logIter = 500;

	CLI::App app;

	bool parse(int argc, char* argv[]){
		string tmp_r;
		string tmp_s, tmp_y;
		string tmp_m;
		app.add_option("-a,--algorithm", alg, "The algorithm to run")->required();
		app.add_option("-p,--parameter", algParam,
			"The parameter of the algorithm, usually the shape of the algorithm")->required();
		// record-file (parameter)
		app.add_option("-r,--record", fnRecord, "The file of the parameter record")->required();
		app.add_flag("-b,--binary", binary, "Whether the record file is binary");
		app.add_option("-l,--linelist", tmp_r, "The Lines of the parameter to use. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		// data-file
		app.add_option("--dataset", dataset, "The dataset type, default: csv");
		app.add_option("--train-part", dataTrainPart, "Load the train part of the dataset");
		app.add_option("-d,--data", fnData, "The data file")->required();
		app.add_option("--skip", tmp_s, "The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_option("-y,--ylist", tmp_y, "The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_flag("-n,--normalize", normalize, "Whether to do data normalization");
		// output
		app.add_option("-m,--pmethod", pmethod, "The method of calculating priority (project or square).");
		app.add_option("-o,--output", fnOutput, "The output priority file")->required();
		app.add_flag("--obinary", outputBinary, "Whether to output in binary");
		app.add_flag("--ofloat", outputFloat, "Whether to output in 32-bit float (when obinary is set)");
		// others
		app.add_flag("--savememory", saveMemory, "Whether to save memory by running slower");
		app.add_option("-w,--thread", nthread, "Number of thread");
		app.add_flag("-c,--resume", resume, "Resume from the last item of output and append it");
		app.add_option("--memory", tmp_m, "Available memory for caching (support k,m,g)");
		app.add_option("--logiter", logIter, "Interval of reporting progress");

		try {
			app.parse(argc, argv);
			rlines = getIntListByRange(tmp_r);
			sort(rlines.begin(), rlines.end());
			idSkip = getIntListByRange(tmp_s);
			idY = getIntListByRange(tmp_y);
			memory = stoulKMG(tmp_m, true);
		} catch(const CLI::ParseError &e) {
			cout << e.what() << endl;
			return false;
		}
		return true;
	}
	void usage(){
		cout << app.help() << endl;
	}
};

struct PriorityCalculator{
	bool init(const string& pmethod, const bool& saveMem, const size_t buffNumberEach){
		if(pmethod == "square"){
			pf = &PriorityCalculator::prioritySquare;
		} else if(pmethod == "project"){
			if(saveMem){
				pf = &PriorityCalculator::priorityProjectSmallSpace;
				bsize = buffNumberEach;
			} else
				pf = &PriorityCalculator::priorityProjectQuick;
		} else
			return false;
		return true;
	}
	vector<double> priority(Model& m, const DataHolder& dh){
		return (this->*pf)(m, dh);
	}
private:
	size_t bsize;
	using pf_t = vector<double>(PriorityCalculator::*)(Model& m, const DataHolder& dh);
	pf_t pf;
	vector<double> prioritySquare(Model& m, const DataHolder& dh){
		vector<double> res;
		for(size_t i = 0; i < dh.size(); ++i){
			auto g = m.gradient(dh.get(i));
			double p = inner_product(g.begin(), g.end(), g.begin(), 0.0);
			res.push_back(p);
		}
		return res;
	}
	vector<double> priorityProjectSmallSpace(Model& m, const DataHolder& dh){
		const size_t n = m.paramWidth();
		vector<double> avg(n, 0.0);
		vector<vector<double>> buffer;
		for(size_t i = 0; i < dh.size(); ++i){
			auto g = m.gradient(dh.get(i));
			for(size_t j = 0; j < n; ++j)
				avg[j] += g[j];
			if(i < bsize) // buffer the first bsize gradients
				buffer.push_back(move(g));
		}
		vector<double> res;
		for(size_t i = 0; i < dh.size(); ++i){
			auto g = i < bsize ? buffer[i] : m.gradient(dh.get(i));
			double p = inner_product(g.begin(), g.end(), avg.begin(), 0.0);
			res.push_back(p);
		}
		return res;
	}
	vector<double> priorityProjectQuick(Model& m, const DataHolder& dh){
		const size_t n = m.paramWidth();
		vector<double> avg(n, 0.0);
		vector<vector<double>> gradient;
		gradient.reserve(dh.size());
		for(size_t i = 0; i < dh.size(); ++i){
			auto g = m.gradient(dh.get(i));
			for(size_t j = 0; j < n; ++j)
				avg[j] += g[j];
			gradient.push_back(move(g));
		}
		for(auto& v : avg)
			v /= gradient.size();
		vector<double> res;
		for(auto& g : gradient){
			double p = inner_product(g.begin(), g.end(), avg.begin(), 0.0);
			res.push_back(p);
		}
		return res;
	}
};

struct PriorityCounter {
	pair<int, ios::streampos> processed(const string& fname, size_t npoint, bool obinary, bool ofloat){
		if(obinary){
			int byte = ofloat ? sizeof(float) : sizeof(double);
			return processedBinary(fname, npoint, byte);
		} else{
			return processedCSV(fname, npoint);
		}
	}
private:
	pair<int, ios::streampos> processedBinary(const string& fname, size_t npoint, int byte){
		ifstream fin(fname, ios::binary);
		if(!fin)
			return make_pair(0, 0);
		fin.seekg(0, ios::end);
		ios::streampos length = fin.tellg();
		size_t n = length / byte / npoint;
		if(n*byte*npoint != length)
			length = n * byte*npoint;
		return make_pair(static_cast<int>(n), length);
	}
	pair<int, ios::streampos> processedCSV(const string& fname, size_t npoint){
		ifstream fin(fname);
		if(!fin)
			return make_pair(0, 0);
		string line;
		int n = 0;
		ios::streampos length;
		while(getline(fin, line)){
			int c = 0;
			for(char ch : line)
				if(c == ',')
					++c;
			if(c != npoint - 1)
				break;
			++n;
			length = fin.tellg();
		}
		return make_pair(n, length);
	}
};

struct PriorityDumper {
	bool init(const string& fname, bool obinary, bool ofloat, bool resume=false, ios::streampos pos=0){
		if(obinary){
			if(ofloat)
				pf = &PriorityDumper::dumpFloat;
			else
				pf = &PriorityDumper::dumpDouble;
		} else{
			pf = &PriorityDumper::dumpCSV;
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
	void dump(const std::vector<double>& priority)
	{
		(this->*pf)(priority);
	}
private:
	ofstream fout;
	using pf_t = void (PriorityDumper::*)(const std::vector<double>&);
	pf_t pf;
	void dumpCSV(const std::vector<double>& priority){
		for(auto& v : priority){
			fout << v << ",";
		}
		fout.seekp(-1, ios::cur);
		fout << "\n";
	}
	void dumpDouble(const std::vector<double>& priority){
		fout.write((const char*)priority.data(), priority.size() * sizeof(double));
	}
	void dumpFloat(const std::vector<double>& priority){
		for(auto& v : priority){
			float f = static_cast<float>(v);
			fout.write((const char*)&f, sizeof(float));
		}
	}
};

int checkRecordLineId(const vector<int>& rlines, const int idx){
	if(rlines.empty() || binary_search(rlines.begin(), rlines.end(), idx)){
		return 1;
	}else{
		if(idx > rlines.back())
			return -1;
		return 0;
	}
}

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);

	DataLoader dl;
	dl.init(opt.dataset, 1, 0, false);
	dl.bindParameter(",", opt.idSkip, opt.idY, false);
	DataHolder dh = dl.load(opt.fnData, opt.dataTrainPart);
	if(opt.normalize)
		dh.normalize(false);

	Model m;
	try{
		m.init(opt.alg, opt.algParam);
	} catch(exception& e){
		cerr << "Error in initialize model" << endl;
		cerr << e.what() << endl;
		return 3;
	}
	if(!m.checkData(dh.xlength(), dh.ylength())){
		cerr << "data size does not match model" << endl;
		return 4;
	}

	ParamArchiver archiver;
	if(!archiver.init_read(opt.fnRecord, m.paramWidth(), opt.binary)){
		cerr << "cannot open record file: " << opt.fnRecord << endl;
		return 5;
	}

	pair<int, ios::streampos> processed(0, 0);
	if(opt.resume){
		PriorityCounter pc;
		processed = pc.processed(opt.fnOutput, dh.size(), opt.outputBinary, opt.outputFloat);
		cout << "Resume after parameter: " << processed.first << ", offset: " << processed.second << endl;
	}

	PriorityDumper dumper;
	if(!dumper.init(opt.fnOutput, opt.outputBinary, opt.outputFloat, opt.resume, processed.second)){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 6;
	}

	PriorityCalculator calculator;
	if(!calculator.init(opt.pmethod, opt.saveMemory,
		opt.memory/(m.paramWidth()*sizeof(double))/opt.nthread))
	{
		cerr << "cannot initialize priority calculator with method: " << opt.pmethod << endl;
		return 7;
	}

	cout << "# of data points: " << dh.size() << ", # of parameters: " << m.paramWidth()
		<< ", required memory (MB): " << ((m.paramWidth() * dh.size() * sizeof(double)) >> 20)
		<< endl;

	int iter;
	double time;
	Parameter param;
	int idx = 0;
	int ndump = 0;
	// skip the processed
	while(ndump < processed.first && !archiver.eof() && archiver.valid()){
		if(!archiver.load(iter, time, param))
			continue;
		int state = checkRecordLineId(opt.rlines, idx++);
		if(state == -1){
			break;
		} else if(state == 0){
			continue;
		} else{
			++ndump;
		}
	}
	// process
	if(opt.nthread == 1){
		while(!archiver.eof() && archiver.valid()){
			if(!archiver.load(iter, time, param))
				continue;
			int state = checkRecordLineId(opt.rlines, idx++);
			if(state == -1){
				break;
			} else if(state == 0){
				continue;
			}
			m.setParameter(move(param));
			vector<double> priority = calculator.priority(m, dh);
			dumper.dump(priority);
			++ndump;
			if(ndump % opt.logIter == 0)
				cout << "  processed: " << ndump << endl;
		}
	} else{
		while(!archiver.eof() && archiver.valid()){
			vector<future<vector<double>>> handlers;
			int i = 0;
			while(i < opt.nthread && !archiver.eof() && archiver.valid()){
				if(!archiver.load(iter, time, param))
					continue;
				int state = checkRecordLineId(opt.rlines, idx++);
				if(state == -1){
					break;
				} else if(state == 0){
					continue;
				}
				m.setParameter(move(param));
				handlers.push_back(async(launch::async, [&](Model m){
					return calculator.priority(m, dh);
				}, m));
				++i;
			}
			for(size_t i = 0; i < handlers.size(); ++i){
				auto priority = handlers[i].get();
				dumper.dump(priority);
				++ndump;
				if(ndump % opt.logIter == 0)
					cout << "  processed: " << ndump << endl;
			}
		}
	}
	archiver.close();
	return 0;
}
