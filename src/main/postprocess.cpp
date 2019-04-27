#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <future>
#include "data/DataHolder.h"
#include "util/Util.h"
#include "model/Model.h"
#include "model/ParamArchiver.h"
#include "func.h"
#include "ParameterIO.h"
#include "CLI11.hpp"

using namespace std;

struct Option {
	int nthread = 1;
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
	//bool iteration = false;
	bool accuracy = false;
	bool show = false;
	size_t topk_data = 0;
	size_t topk_param = 0;

	CLI::App app;

	bool parse(int argc, char* argv[]){
		string tmp_s, tmp_y;
		app.add_option("-w,--thread", nthread, "Number of thread");
		app.add_option("-a,--algorithm", alg, "The algorithm to run")->required();
		app.add_option("-p,--parameter", algParam,
			"The parameter of the algorithm, usually the shape of the algorithm")->required();
		app.add_option("-r,--record", fnRecord, "The file of the parameter record")->required();
		app.add_flag("-b,--binary", binary, "Whether the record file is binary");
		app.add_option("--top-param", topk_data, "Only use the top-k parameters");
		// data-file
		app.add_option("-d,--data", fnData, "The data file")->required();
		app.add_option("--skip", tmp_s, "The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_option("-y,--ylist", tmp_y, "The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)");
		app.add_flag("-n,--normalize", normalize, "Whether to do data normalization");
		app.add_option("--top-data", topk_data, "Only use the top-k data points");
		//
		app.add_option("--reference", fnParam, "The referenced parameter file");
		// output
		app.add_option("-o,--output", fnOutput, "The output file");
		app.add_flag("--accuracy", accuracy, "Calculate the accuracy");
		//app.add_flag("--iteration", iteration, "Use the iteration as the first column");
		app.add_flag("--show", show, "Show the result on STDOUT");

		try {
			app.parse(argc, argv);
			if(nthread <= 0)
				nthread = 1;
			idSkip = getIntListByRange(tmp_s);
			idY = getIntListByRange(tmp_y);
			auto it = remove_if(idY.begin(), idY.end(), [](const int v){
				return v < 0;
			});
			idY.erase(it, idY.end());
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

struct TaskResult{
	double loss;
	double diff;
	size_t correct;
};

TaskResult evaluateOne(Parameter& param, Model& m,
	const bool withRef, const bool doAccuracy,
	const DataHolder& dh, const vector<double>& ref)
{
	double diff = 0.0;
	if(withRef)
		diff = vectorDifference(ref, param.weights);
	m.setParameter(param);
	double loss = 0.0;
	size_t correct = 0;
	for(size_t i = 0; i < dh.size(); ++i){
		auto& d = dh.get(i);
		auto p = m.predict(d);
		loss += m.loss(p, d.y);
		if(!doAccuracy)
			continue;
		for(size_t j = 0; j < p.size(); ++j)
			if(m.classify(p[j]) == d.y[j])
				++correct;
	}
	loss /= dh.size();
	return TaskResult{ loss, diff, correct };
}

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);

	string algParam = opt.algParam;
	vector<double> reference;
	const bool withRef = !opt.fnParam.empty();
	if(withRef){
		ifstream finr(opt.fnParam);
		if(finr.fail()){
			cerr << "cannot open reference parameter file: " << opt.fnParam << endl;
			return 5;
		}
		ParameterIO io(opt.alg, "");
		string tmp;
		tie(tmp, reference) = io.load(finr);
		if(tmp != algParam){
			cerr << "Warning: given parameter does not match the one read from parameter file" << endl;
		}
		algParam = tmp;
	}
	const bool doAccuracy = opt.accuracy && !opt.idY.empty();

	ofstream fout(opt.fnOutput);
	if(!opt.fnOutput.empty() && fout.fail()){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 2;
	}
	const bool write = !opt.fnOutput.empty();

	DataHolder dh(1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, false, true, opt.topk_data);
	if(opt.normalize)
		dh.normalize(false);
	double accuracy_factor = 0.0;
	if(dh.size() != 0 && dh.ylength() != 0)
		accuracy_factor = 1.0 / dh.size() / dh.ylength();

	Model m;
	try{
		m.init(opt.alg, algParam);
		m.checkData(dh.xlength(), dh.ylength());
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

	if(opt.nthread == 1){ // 1-thread
		vector<double> last(dh.xlength(), 0.0);
		int iter;
		double time;
		Parameter param;
		int idx = 0;
		while(!archiver.eof() && archiver.valid()){
			if(!archiver.load(iter, time, param))
				continue;
			if(!opt.show && idx % 500 == 0)
				cout << "  processed: " << idx << endl;
			if(opt.topk_param != 0 && idx >= opt.topk_param)
				break;
			++idx;
			TaskResult tr = evaluateOne(param, m, withRef, doAccuracy, dh, reference);
			double impro = vectorDifference(last, param.weights);
			last = move(param.weights);
			double accuracy = tr.correct * accuracy_factor;
			if(opt.show){
				cout << showpoint << iter << "\t" << time << "\t" << tr.loss << "\t"
					<< noshowpoint << accuracy << "\t" << tr.diff << "\t" << impro << endl;
			}
			if(write){
				fout << showpoint << iter << "," << time << "," << tr.loss << ","
					<< noshowpoint << accuracy << "," << tr.diff << "," << impro << "\n";
			}
		}
	} else{ // n-thread
		vector<double> last(dh.xlength(), 0.0);
		vector<Model> models(opt.nthread);
		for(size_t i = 0; i < opt.nthread; ++i)
			models[i].init(opt.alg, opt.algParam);
		vector<tuple<int, double, Parameter>> data(opt.nthread);
		vector<double> improvements(opt.nthread);
		int idx = 0;
		while(!archiver.eof() && archiver.valid()){
			vector<future<TaskResult>> handlers;
			int i = 0;
			while(i < opt.nthread && !archiver.eof() && archiver.valid()){
				if(!archiver.load(get<0>(data[i]), get<1>(data[i]), get<2>(data[i])))
					continue;
				if(!opt.show && idx % 500 == 0)
					cout << "  processed: " << idx << endl;
				if(opt.topk_param != 0 && idx >= opt.topk_param)
					break;
				auto& p = get<2>(data[i]);
				handlers.push_back(async(launch::async, &evaluateOne,
					ref(p), ref(models[i]), withRef, doAccuracy, cref(dh), cref(reference)));
				++idx;
				++i;
			}
			for(size_t i = 0; i < handlers.size(); ++i){
				auto& p = get<2>(data[i]);
				improvements[i] = vectorDifference(last, p.weights);
				last = p.weights;
			}
			for(size_t i = 0; i < handlers.size(); ++i){
				int iter = get<0>(data[i]);
				double time = get<1>(data[i]);
				TaskResult tr = handlers[i].get();
				double accuracy = tr.correct * accuracy_factor;
				if(opt.show){
					cout << showpoint << iter << "\t" << time << "\t" << tr.loss << "\t"
						<< noshowpoint << accuracy << "\t" << tr.diff << "\t" << improvements[i] << endl;
				}
				if(write){
					fout << showpoint << iter << "," << time << "," << tr.loss << ","
						<< noshowpoint << accuracy << "," << tr.diff << "," << improvements[i] << "\n";
				}
			}
		}
		for(size_t i = 0; i < opt.nthread; ++i)
			models[i].clear();
	}
	m.clear();
	archiver.close();
	fout.close();
	return 0;
}
