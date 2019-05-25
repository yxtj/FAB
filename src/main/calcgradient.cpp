#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "data/DataHolder.h"
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
	vector<int> idSkip;
	vector<int> idY;
	bool normalize = false;

	string fnOutput;
	bool outputBinary = false;
	bool outputFloat = true;

	CLI::App app;

	bool parse(int argc, char* argv[]){
		string tmp_r;
		string tmp_s, tmp_y;
		app.add_option("-a,--algorithm", alg, "The algorithm to run")->required();
		app.add_option("-p,--parameter", algParam,
			"The parameter of the algorithm, usually the shape of the algorithm")->required();
		app.add_option("-r,--record", fnRecord, "The file of the parameter record")->required();
		app.add_flag("-b,--binary", binary, "Whether the record file is binary");
		app.add_option("-l,--linelist", tmp_r, "The Lines of the parameter to use. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")->required();
		// data-file
		app.add_option("-d,--data", fnData, "The data file")->required();
		app.add_option("--skip", tmp_s, "The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_option("-y,--ylist", tmp_y, "The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_flag("-n,--normalize", normalize, "Whether to do data normalization");
		// output
		app.add_option("-o,--output", fnOutput, "The output gradient file")->required();
		app.add_flag("--obinary", outputBinary, "Whether to output in binary");
		app.add_flag("--ofloat", outputFloat, "Whether to output in 32-bit float (when obinary is set)");

		try {
			app.parse(argc, argv);
			rlines = getIntListByRange(tmp_r);
			sort(rlines.begin(), rlines.end());
			idSkip = getIntListByRange(tmp_s);
			idY = getIntListByRange(tmp_y);
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

struct GradientDumper{
	bool init(const string& fname, bool obinary, bool ofloat){
		if(obinary){
			if(ofloat)
				pf = &GradientDumper::dumpFloat;
			else
				pf = &GradientDumper::dumpDouble;
		} else{
			pf = &GradientDumper::dumpCSV;
		}
		ios_base::openmode f = ios::out;
		if(obinary)
			f |= ios::binary;
		fout.open(fname, f);
		return fout.is_open();
	}
	void dump(const std::vector<std::vector<double>>& gradient)
	{
		(this->*pf)(gradient);
	}
private:
	ofstream fout;
	using pf_t = void (GradientDumper::*)(const std::vector<std::vector<double>>&);
	pf_t pf;
	void dumpCSV(const std::vector<std::vector<double>>& gradient){
		for(auto& line : gradient){
			for(auto& v : line){
				fout << v << ",";
			}
			fout.seekp(-1, ios::cur);
			fout << "\n";
		}
	}
	void dumpDouble(const std::vector<std::vector<double>>& gradient){
		for(auto& line : gradient){
			fout.write((const char*)line.data(), line.size() * sizeof(double));
		}
	}
	void dumpFloat(const std::vector<std::vector<double>>& gradient){
		for(auto& line : gradient){
			for(auto& v : line){
				float f = static_cast<float>(v);
				fout.write((const char*)&f, sizeof(float));
			}
		}
	}
};

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);

	ofstream fout(opt.fnOutput);
	if(!opt.fnOutput.empty() && fout.fail()){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 2;
	}
	const bool write = !opt.fnOutput.empty();

	DataHolder dh(1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, false, true);
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

	GradientDumper dumper;
	if(!dumper.init(opt.fnOutput, opt.outputBinary, opt.outputFloat)){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 6;
	}

	cout << "# of data points: " << dh.size() << ", # of parameters: " << m.paramWidth() << endl;

	int iter;
	double time;
	Parameter param;
	int idx = 0;
	while(!archiver.eof() && archiver.valid()){
		if(!archiver.load(iter, time, param))
			continue;
		if(!binary_search(opt.rlines.begin(), opt.rlines.end(), idx++)){
			if(idx > opt.rlines.back())
				break;
			continue;
		}
		m.setParameter(move(param));
		vector<vector<double>> gradient;
		gradient.reserve(dh.size());
		for(size_t i = 0; i < dh.size(); ++i){
			auto g = m.gradient(dh.get(i));
			gradient.push_back(move(g));
		}
		dumper.dump(gradient);
	}
	archiver.close();
	fout.close();
	return 0;
}
