#include "Option.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include "util/Util.h"

using namespace std;

struct Option::Impl{
	boost::program_options::options_description desc;
	Impl(const int width):desc("Options", width)
	{}
};

bool Option::parse(int argc, char * argv[], const size_t nWorker)
{
	string tmp_cast;
	string tmp_interval;
	string tmp_ids, tmp_idy;
	string tmp_bs;
	string tmp_t_iter, tmp_a_iter, tmp_l_iter;
	int tmp_v;
	if(pimpl == nullptr)
		pimpl = new Impl(getScreenSize().first);
	using boost::program_options::value;
	using boost::program_options::bool_switch;
	pimpl->desc.add_options()
		("help,h", "Print help messages")
		// parallel
		("mode,m", value(&mode)->required(), "The parallel mode: bsp, tap, ssp:<n>, sap:<n>, fsp, aap")
		// parallel - broadcast
		("cast_mode,c", value(&tmp_cast)->default_value("broadcast"),
			"The method to send out new parameters. Supports: broadcast/all, ring:k, random:k,seed, hash:k")
		// parallel - fsp
		("flex_interval", value(&tmp_interval)->default_value("portion:0.05"),
			"The method to decide update interval for FSP. "
			"Supports: interval:x(x is in seconds), portion : x(x in 0~1), "
			"improve:x,t (x: avg. imporovement, t: max waiting time), balance:w (num. of windows)")
		// app - algorithm
		("algorithm,a", value(&algorighm)->required(), "The algorithm to run. "
			"Support: lr, mlp, cnn, rnn, tm, kmeans.")
		("parameter,p", value(&algParam)->required(),
			"The parameter of the algorithm, usually the shape of the algorithm")
		("seed", value(&seed)->default_value(123456U), "The seed to initialize parameters")
		// app - training
		("batch_size,s", value(&tmp_bs)->required(), "The global batch size. Support suffix: k, m, g")
		//("learning_rate,l", value(&lrate)->required(), "The learning rate")
		("optimizer,o", value(&optimizer)->required()->default_value("gd:0.01"),
			"The optimizer to train. Support: gd:<lr>, em:<lr>.")
		// file - input
		("data_file,d", value(&fnData)->required(), "The file name of the input data")
		("skip", value(&tmp_ids)->default_value({}, ""),
			"The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		("ylist,y", value(&tmp_idy)->default_value({}, ""),
			"The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		("normalize,n", bool_switch(&normalize)->default_value(false),
			"Whether to do normailzation on the input file")
		// file - output
		("record_file,r", value(&fnOutput)->required(), "The file name of the archived parameter")
		("binary,b", bool_switch(&binary)->default_value(false), "Whether to output using binary IO")
		// termination
		("term_iter", value(&tmp_t_iter)->required(), "Termination condition: maximum iteration")
		("term_time", value(&tcTime)->required(), "Termination condition: maximum training time")
		// archive
		("arch_iter", value(&tmp_a_iter)->default_value("1000"), "Progress archiving condition: maximum iteration")
		("arch_time", value(&arvTime)->default_value(1.0), "Progress archiving condition: maximum training time")
		// log
		("log_iter", value(&tmp_l_iter)->default_value("100"), "Log training step every <log_iter> iterations")
		("v", value(&tmp_v), "Verbose level")
		;

	boost::program_options::variables_map vm;
	this->nw = nWorker;
	try {
		auto p = boost::program_options::command_line_parser(argc, argv).
			//options(pimpl->desc).allow_unregistered().run();
			options(pimpl->desc).run();
		boost::program_options::store(p, vm);
		boost::program_options::notify(vm);

		mcastParam = getStringList(tmp_cast, ":,; ");
		intervalParam = getStringList(tmp_interval, ":,; ");
		idSkip = getIntListByRange(tmp_ids);
		idY = getIntListByRange(tmp_idy);
		batchSize = stoiKMG(tmp_bs);
		tcIter = stoiKMG(tmp_t_iter);
		arvIter = stoiKMG(tmp_a_iter);
		logIter = stoiKMG(tmp_l_iter);
	} catch(exception& e){
		cerr << "Error in parsing parameter: " << e.what() << endl;
		return false;
	}
	if(vm.count("help") != 0)
		return false;

	// check and preprocess
	if(!preprocessMode()){
		cerr << "Error: mode not supported: " << mode << endl;
		return false;
	}
	if(!processAlgorithm()){
		cerr << "Error: algorithm not supported: " << algorighm << endl;
		return false;
	}
	if(!processOptimizer()){
		cerr << "Error: optimizer not supported: " << optimizer << endl;
		return false;
	}
	// error handling

	return true;
}

void Option::showUsage() const {
	if(pimpl == nullptr)
		return;
	cout << pimpl->desc << endl;
	/*
	cout << "usage: <mode> <alg> <param> <data-file> <output-file> <id-skip> <id-y> <normalize>"
		" <lrate> <batch-size> <term-iter> <term-time> [arv-iter=1000] [arv-time=0.5] [log-iter=1000]"
		" [flex-param=portion:0.05] [mcast-param=all]" << endl;
	//cout << "usage: <algorithm> <mode> <data-file> <output-file> <id-skip> <id-y> <nw> <batch-size> <term-iter> <term-time>" << endl;
	cout << "  <mode>: bsp, tap, ssp:<n>, fsp, aap\n"
		<< "  <alg>: algorithm name. Support: lr, mlp, cnn, rnn, tm.\n"
		<< "  <param>: parameter of the algorithm, usually the shape of the algorithm.\n"
		<< "  <id-skip> <id-y>: a list separated with space, comma and a-b (a, a+1, a+2, ..., b).\n"
		<< "  [flex-param]: supports: interval:x (x is in seconds), portion:x (x in 0~1), "
		"improve:x,t (x: avg. imporovement, t: max waiting time), balance:w (num. of windows)\n"
		<< "  [mcast-param]: supports: all, ring:k, random:k,seed, hash:k\n"
		<< endl;
	*/
}

bool Option::preprocessMode(){
	for(char& ch : mode){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(mode, ":-, ");
	vector<string> supported = { "bsp", "tap", "ssp", "sap", "fsp", "aap" };
	auto it = find(supported.begin(), supported.end(), t[0]);
	if(it == supported.end())
		return false;
	mode = t[0];
	if(t[0] == "ssp" || t[0] == "sap"){
		if(t.size() > 1)
			staleGap = stoi(t[1]);
		else
			staleGap = 1;
	} else if(t[0]=="aap"){
		aapWait = t.size() >= 2 && beTrueOption(t[1]);
	}
	return true;
}

bool Option::processAlgorithm(){
	for(char& ch : algorighm){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> supported = { "lr", "mlp", "cnn", "rnn", "tm", "kmeans" };
	auto it = find(supported.begin(), supported.end(), algorighm);
	return it != supported.end();
}

bool Option::processOptimizer()
{
	for(char& ch : optimizer){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(optimizer, ":-, ");
	vector<string> supported = { "gd", "em" };
	auto it = find(supported.begin(), supported.end(), t[0]);
	if(it == supported.end())
		return false;
	optimizer = t[0];
	for(size_t i = 1; i < t.size(); ++i)
		optimizerParam.push_back(t[i]);
	return true;
}

