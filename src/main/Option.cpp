#include "Option.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>
#include <boost/program_options.hpp>
#include "util/Util.h"
#include "data/DataLoader.h"
#include "model/KernelFactory.h"
#include "train/TrainerFactory.h"

using namespace std;

struct Option::Impl{
	boost::program_options::options_description desc;
	Impl(const int width):desc("Options", width)
	{}
};

static string VecToString(const vector<string>& vec){
	string res;
	for(auto& v : vec)
		res += v + " ";
	if(!vec.empty())
		res.pop_back();
	return res;
}

bool Option::parse(int argc, char * argv[], const size_t nWorker)
{
	string tmp_cast;
	string tmp_interval;
	string tmp_ids, tmp_idy;
	string tmp_bs, tmp_rs;
	string tmp_sr, tmp_sh;
	string tmp_t_point, tmp_t_delta, tmp_t_iter;
	string tmp_a_iter, tmp_l_iter;
	int tmp_v;
	if(pimpl == nullptr)
		pimpl = new Impl(getScreenSize().first);

	string desc_alg = "The algorithm/model to run/train. "
		"Support: " + VecToString(KernelFactory::supportList());
	string desc_opt = "The optimizer to adopt. "
		"Support: " + VecToString(TrainerFactory::supportList());
	string desc_dl = "The dataset/type to use. Give dataset name (mnist) or "
		"type (csv, tsv, customize, list (variable length x)). "
		"Support: " + VecToString(DataLoader::supportList());;

	using boost::program_options::value;
	using boost::program_options::bool_switch;
	pimpl->desc.add_options()
		("help,h", "Print help messages.")
		// parallel
		("mode,m", value(&conf.mode)->default_value("bsp"), "The parallel mode: bsp, tap, ssp:<n>, sap:<n>, fsp, aap, pap:<p>:<d>.")
		// parallel - broadcast
		("cast_mode,c", value(&tmp_cast)->default_value("broadcast"),
			"The method to send out new parameters. Supports: broadcast/all, ring:k, random:k,seed, hash:k.")
		// parallel - fsp
		("flex_interval", value(&tmp_interval)->default_value("portion:0.05"),
			"The method to decide update interval for FSP. "
			"Supports: interval:x(x is in seconds), portion:x(x in 0~1), "
			"improve:x,t (x: avg. imporovement, t: max waiting time), balance:w (num. of windows).")
		// parallel - probe
		("probe", bool_switch(&conf.probe)->default_value(false), "Probe the best hyper-parameter like global batch size.")
		("probe_ratio", value(&conf.probeRatio)->default_value(0.05), "The ratio of data used for each probe.")
		("probe_loss_online", bool_switch(&conf.probeOnlineLoss)->default_value(false), 
			"Use the accumulated online loss or calculate loss with last model parameter.")
		("probe_loss_full", bool_switch(&conf.probeLossFull)->default_value(false),
			"Calcualte the loss with full batch or the probed part.")
		// setting - machine speed
		("speed_random", value(&tmp_sr)->default_value(""),
			"The random worker speed difference (how much slower than normal).\n"
			"Format: <type>:<min>:<max>:<distribution parameters...>. "
			"<type> supports: <empty>, none, exp, norm.")
		("speed_hetero", value(&tmp_sh)->default_value(""), "The fixed worker speed difference (how much slower than normal).\n"
			"Format: <n1>-<m1>:<p1>,<n2>-<m2>:<p2>,<n3>:<p3>,... "
			"<n1>-<m1>:<p1> means worker n1,n1+1,...,m1 are p1 slower than normal. The abscent workers are assumed 0 (working normally).")
		// app - algorithm
		("algorithm,a", value(&conf.algorighm)->required(), desc_alg.c_str())
		("parameter,p", value(&conf.algParam)->required(),
			"The parameter of the algorithm, usually the shape of the algorithm.")
		("seed", value(&conf.seed)->default_value(123456U), "The seed to initialize parameters.")
		// app - training
		("batch_size,s", value(&tmp_bs)->required(), "The global batch size. Support suffix: k, m, g.")
		("report_size", value(&tmp_rs)->default_value("1"), "The local report size. Support suffix: k, m, g.")
		("optimizer,o", value(&conf.optimizer)->required()->default_value("gd:0.01"), desc_opt.c_str())
		// file - input
		("dataset", value(&conf.dataset)->default_value("csv"), desc_dl.c_str())
		("trainpart", bool_switch(&conf.trainPart)->default_value(true),
			"Whether to use the trainning part of the dateset.")
		("data_file,d", value(&conf.fnData)->required(), "The file name of the input data.")
		("topk,k", value(&conf.topk)->default_value(0), "Only use the first <k> data points (0 means all).")
		("normalize,n", bool_switch(&conf.normalize)->default_value(false),
			"Whether to do normailzation on the input file.")
		("shuffle", bool_switch(&conf.shuffle)->default_value(false), "Randomly shuffle the dataset.")
		// file - input - table
		("header", bool_switch(&conf.header)->default_value(false), 
			"Whether the input file contain a header line")
		("sepper", value(&conf.sepper)->default_value(","), "Separator for customized dataset.")
		("skip", value(&tmp_ids)->default_value({}, ""),
			"The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		// file - input - list (variable-length x)
		("unit", value(&conf.lenUnit)->default_value(0), "Length of one x unit for the variable length input (RNN).")
		("ylist,y", value(&tmp_idy)->default_value({}, ""),
			"The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		// file - output
		("record_file,r", value(&conf.fnOutput)->required(), "The file name of the archived parameter.")
		("binary,b", bool_switch(&conf.binary)->default_value(false), "Whether to output using binary IO.")
		("resume", bool_switch(&conf.resume)->default_value(false), "Whether to resume from the last output item.")
		// termination
		("term_point", value(&tmp_t_point)->default_value("0"), "Termination condition: maximum used data point.")
		("term_delta", value(&tmp_t_delta)->default_value("0"), "Termination condition: maximum delta report.")
		("term_iter", value(&tmp_t_iter)->default_value("0"), "Termination condition: maximum iteration.")
		("term_time", value(&conf.tcTime)->required(), "Termination condition: maximum training time.")
		// archive
		("arch_iter", value(&tmp_a_iter)->default_value("1000"), "Progress archiving condition: maximum iteration.")
		("arch_time", value(&conf.arvTime)->default_value(1.0), "Progress archiving condition: maximum training time.")
		// log
		("log_iter", value(&tmp_l_iter)->default_value("100"), "Log training step every <log_iter> iterations.")
		("v", value(&tmp_v), "Verbose level.")
		;

	boost::program_options::variables_map vm;
	conf.nw = nWorker;
	try {
		auto p = boost::program_options::command_line_parser(argc, argv).
			//options(pimpl->desc).allow_unregistered().run();
			options(pimpl->desc).run();
		boost::program_options::store(p, vm);
		boost::program_options::notify(vm);

		conf.mcastParam = getStringList(tmp_cast, ":,; ");
		conf.intervalParam = getStringList(tmp_interval, ":,; ");
		conf.idSkip = getIntListByRange(tmp_ids);
		conf.idY = getIntListByRange(tmp_idy);
		conf.batchSize = stoiKMG(tmp_bs);
		conf.reportSize = stoiKMG(tmp_rs);
		if(conf.reportSize == 0)
			conf.reportSize = conf.batchSize / conf.nw;
		conf.tcPoint = stoiKMG(tmp_t_point);
		conf.tcDelta = stoiKMG(tmp_t_delta);
		conf.tcIter = stoiKMG(tmp_t_iter);
		conf.arvIter = stoiKMG(tmp_a_iter);
		conf.logIter = stoiKMG(tmp_l_iter);
	} catch(exception& e){
		cerr << "Error in parsing parameter: " << e.what() << endl;
		return false;
	}
	if(vm.count("help") != 0)
		return false;

	// check and preprocess
	if(!processMode()){
		cerr << "Error: mode not supported: " << conf.mode << endl;
		return false;
	}
	if(!processDataset()){
		cerr << "Error: dataset not supported: " << conf.dataset << endl;
		return false;
	}
	if(!processAlgorithm()){
		cerr << "Error: algorithm not supported: " << conf.algorighm << endl;
		return false;
	}
	if(!processOptimizer()){
		cerr << "Error: optimizer not supported: " << conf.optimizer << endl;
		return false;
	}
	if(!processSpeedRandom(tmp_sr)){
		cerr << "Error: speed adjustment (random) not supported: " << tmp_sr << endl;
		return false;
	}
	if(!processSpeedHeterogenerity(tmp_sh)){
		cerr << "Error: speed adjustment (heterogenerity) not supported: " << tmp_sh << endl;
		return false;
	}
	// error handling

	return true;
}

void Option::showUsage() const {
	if(pimpl == nullptr)
		return;
	cout << pimpl->desc << endl;
}

bool Option::processMode(){
	for(char& ch : conf.mode){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(conf.mode, ":-, ");
	vector<string> supported = { "bsp", "tap", "ssp", "sap", "fsp", "aap", "pap", "pap2" };
	auto it = find(supported.begin(), supported.end(), t[0]);
	if(it == supported.end())
		return false;
	conf.mode = t[0];
	if(t[0] == "ssp" || t[0] == "sap"){
		if(t.size() > 1)
			conf.staleGap = stoi(t[1]);
		else
			conf.staleGap = 1;
	} else if(t[0]=="aap"){
		conf.aapWait = t.size() >= 2 && beTrueOption(t[1]);
	} else if(t[0] == "pap" || t[0] == "pap2"){
		conf.papOnlineProbeVersion = t.size() > 1 ? stoi(t[1]) : false;
		conf.papDynamicBatchSize = t.size() > 2 ? beTrueOption(t[2]) : false;
		conf.papDynamicReportFreq = t.size() > 3 ? beTrueOption(t[3]) : false;
	}
	return true;
}

bool Option::processDataset(){
	for(char& ch : conf.dataset){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	return DataLoader::isSupported(conf.dataset);
}

bool Option::processAlgorithm(){
	for(char& ch : conf.algorighm){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	return KernelFactory::isSupported(conf.algorighm);
}

bool Option::processOptimizer()
{
	for(char& ch : conf.optimizer){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(conf.optimizer, ":;, ");
	if(t.empty() || !TrainerFactory::isSupported(t[0]))
		return false;
	conf.optimizer = t[0];
	for(size_t i = 1; i < t.size(); ++i)
		conf.optimizerParam.push_back(t[i]);
	return true;
}

bool Option::processSpeedRandom(const std::string& srandom)
{
	conf.adjustSpeedRandom = false;
	if(srandom.empty())
		return true;
	vector<string> vec = getStringList(srandom, ":;, ");
	for(string& str : vec){
		for(char& ch : str){
			if(ch >= 'A' && ch <= 'Z')
				ch += 'a' - 'A';
		}
	}
	if(vec[0] == "none")
		return true;
	vector<pair<string, int>> supported = { {"exp", 1}, {"norm", 2}, {"uni", 2} };
	for(auto& p : supported){
		if(vec[0] == p.first){
			if(vec.size() != static_cast<size_t>(3 + p.second)){
				cerr << "Error in parsing speed randomness parameter for: " << p.first << endl;
				return false;
			}
			conf.speedRandomParam.push_back(vec[0]);
			conf.adjustSpeedRandom = true;
			for(size_t i = 3; i < vec.size(); ++i)
				conf.speedRandomParam.push_back(vec[i]);
		}
	}
	if(conf.adjustSpeedRandom == false){
		return false;
	}else{
		try{
			conf.speedRandomMin = stod(vec[1]);
			conf.speedRandomMax = stod(vec[2]);
		} catch(exception& e){
			cerr << "Error in parsing speed randomness bound: " << endl;
			return false;
		}
	}
	return true;
}

bool Option::processSpeedHeterogenerity(const std::string& shetero)
{
	conf.adjustSpeedHetero = false;
	conf.speedHeterogenerity.assign(conf.nw, 0);
	if(!shetero.empty()){
		conf.adjustSpeedHetero = true;
		vector<string> vec= getStringList(shetero, ", ");
		regex reg1("(\\d+)\\:(\\d*.?\\d+)");
		regex reg2("(\\d+)-(\\d+)\\:(\\d*.?\\d+)");
		smatch m;
		for(string& s : vec){
			if(regex_match(s, m, reg1)){
				int f = stoi(m[1]);
				conf.speedHeterogenerity[f] = stod(m[2]);
			} else if(regex_match(s, m, reg2)){
				int f = stoi(m[1]);
				int l = stoi(m[2]);
				double v = stod(m[3]);
				for(; f <= l; ++f)
					conf.speedHeterogenerity[f] = v;
			} else{
				return false;
			}
		}
	}
	return true;
}

