#include <string>
#include <vector>
#include <iostream>
#include <fstream>
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
	string fnData;
	vector<int> idSkip;
	vector<int> idY;
	string fnParam;
	string fnOutput;
	bool normalize = false;
	bool binary = false;
	bool accuracy = false;
	bool show = false;
	size_t topk = 0;

	CLI::App app;

	bool parse(int argc, char* argv[]){
		string tmp_s, tmp_y;
		app.add_option("-a,--algorithm", alg, "The algorithm to run")->required();
		app.add_option("-p,--parameter", algParam,
			"The parameter of the algorithm, usually the shape of the algorithm")->required();
		app.add_option("-r,--record", fnRecord, "The file of the parameter record")->required();
		// data-file
		app.add_option("-d,--data", fnData, "The data file")->required();
		app.add_option("--skip", tmp_s, "The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_option("--ylist", tmp_y, "The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_flag("-n,--normalize", normalize, "Whether to do data normalization");
		app.add_flag("-b,--binary", binary, "Whether to do data normalization");
		app.add_option("-k,--topk", topk, "Only use the top-k data points");
		//
		app.add_option("--reference", fnParam, "The referenced parameter file");
		// output
		app.add_option("-o,--output", fnOutput, "The output file");
		app.add_flag("--accuracy", accuracy, "Show the accuracy");
		app.add_flag("--show", show, "Show the result on STDOUT");

		try {
			app.parse(argc, argv);
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

struct ParameterLoader{
	pair<string, vector<double>> loadParameter(const string& name, ifstream& fin){
		if(name == "lr")
			return funLR(fin);
		else
			return funGeneral(fin);
		return {};
	}
private:
	pair<string, vector<double>> funLR(ifstream& fin){
		string line;
		getline(fin, line);
		vector<double> vec = getDoubleList(line);
		string param = to_string(vec.size() - 1);
		return make_pair(move(param), move(vec));
	}
	pair<string, vector<double>> funGeneral(ifstream& fin){
		string line;
		getline(fin, line);
		string param = line;
		vector<int> shape = getIntList(line, " ,-");
		vector<double> vec;
		for(size_t i = 0; i < shape.size() - 1; ++i){
			getline(fin, line);
			vector<double> temp = getDoubleList(line);
			vec.insert(vec.end(), temp.begin(), temp.end());
		}
		return make_pair(move(param), move(vec));
	}
};

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);

	string algParam = opt.algParam;
	vector<double> ref;
	const bool withRef = !opt.fnParam.empty();
	if(withRef){
		ifstream finr(opt.fnParam);
		if(finr.fail()){
			cerr << "cannot open reference parameter file: " << opt.fnParam << endl;
			return 5;
		}
		ParameterIO io(opt.alg, "");
		string tmp;
		tie(tmp, ref) = io.load(finr);
		if(tmp != algParam){
			cerr << "Warning: given parameter does not match the one read from parameter file" << endl;
		}
		algParam = tmp;
	}

	ofstream fout(opt.fnOutput);
	if(!opt.fnOutput.empty() && fout.fail()){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 2;
	}
	const bool write = !opt.fnOutput.empty();

	DataHolder dh(false, 1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, false, true, opt.topk);
	if(opt.normalize)
		dh.normalize(false);
	double accuracy_factor = 1.0 / dh.size() / dh.ylength();

	Model m;
	try{
		m.init(opt.alg, dh.xlength(), algParam);
	} catch(exception& e){
		cerr << "Error in initialize model" << endl;
		cerr << e.what() << endl;
		return 3;
	}

	ParamArchiver archiver;
	if(!archiver.init_read(opt.fnRecord, m.paramWidth(), opt.binary)){
		cerr << "cannot open record file: " << opt.fnRecord << endl;
		return 4;
	}

	vector<double> last(dh.xlength(), 0.0);
	int iter;
	double time;
	Parameter param;
	int idx = 0;
	while(!archiver.eof()){
		if(!archiver.load(iter, time, param))
			continue;
		if(idx++ % 10000 == 0)
			cout << "  processed: " << idx << endl;
		//if(idx++ < 500)
		//	continue;
		double diff = 0.0;
		if(withRef)
			diff = vectorDifference(ref, param.weights);
		double impro = vectorDifference(last, param.weights);
		last = param.weights;
		double loss = 0.0;
		size_t correct = 0;
		for(size_t i = 0; i < dh.size(); ++i){
			auto& d = dh.get(i);
			auto p = m.predict(d);
			loss += m.loss(p, d.y);
			if(!opt.accuracy)
				continue;
			for(size_t j = 0; j < p.size(); ++j)
				if(m.classify(p[j]) == d.y[j])
					++correct;
		}
		loss /= dh.size();
		double accuracy = correct * accuracy_factor;
		if(opt.show)
			cout << time << "\t" << loss << "\t" << accuracy << "\t" << diff << "\t" << impro << endl;
		if(write)
			fout << time << "," << loss << "," << accuracy << "," << diff << "," << impro << "\n";
	}
	archiver.close();
	fout.close();
	return 0;
}
