#include "Option.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include "util/Util.h"

using namespace std;

bool Option::parse(int argc, char * argv[], const size_t nWorker)
{
	this->nw = nWorker;
	//return dummySet(*this, argc, argv);

	int idx = 1;
	int optIdx = 13;
	bool flag = false;
	// parse
	if(argc < optIdx)
		return false;
	try{
		mode = argv[idx++];
		algorighm = argv[idx++];
		algParam = argv[idx++];
		aapWait = false;

		fnData = argv[idx++];
		fnOutput = argv[idx++];
		idSkip = getIntList(argv[idx++]);
		idY = getIntList(argv[idx++]);

		doNormalize = beTrueOption(argv[idx++]);
		lrate = stod(argv[idx++]);
		batchSize = stoulKMG(argv[idx++]);
		// finish condition
		tcIter = stoulKMG(argv[idx++]); // maximum iteration
		tcTime = stod(argv[idx++]); // maximum training time
		//tcDiff = stod(argv[idx++]); // minimum improvement cross iterations

		arvIter = argc > optIdx++ ? stoiKMG(argv[idx++]) : 1000;
		arvTime = argc > optIdx++ ? stod(argv[idx++]) : 0.5;
		logIter = argc > optIdx++ ? stoiKMG(argv[idx++]) : 1000;
		string itvparam = argc > optIdx++ ? argv[idx++] : "portion:0.05";
		intervalParam = getStringList(itvparam, ":,; ");
		string mparam = argc > optIdx++ ? argv[idx++] : "all";
		mcastParam = getStringList(mparam, ":,; ");
	} catch(exception& e){
		cerr << "Cannot parse the " << idx << "-th parameter: " << argv[idx] << endl;
		cerr << "Error message: " << e.what() << endl;
		return false;
	}
	// check and preprocess
	if(!preprocessMode()){
		cerr << "Error: mode not supported: " << mode << endl;
		return false;
	}
	if(!processAlgorithm()){
		cerr << "Error: algorithm not supported: " << algorighm << endl;
		return false;
	}
	// error handling

	return true;
}

void Option::showUsage() const {
	cout << "usage: <mode> <alg> <param> <data-file> <output-file> <id-skip> <id-y> <normalize>"
		" <lrate> <batch-size> <term-iter> <term-time> [arv-iter=1000] [arv-time=0.5] [log-iter=1000]"
		" [flex-param=portion:0.05] [mcast-param=all]" << endl;
	//cout << "usage: <algorithm> <mode> <data-file> <output-file> <id-skip> <id-y> <nw> <batch-size> <term-iter> <term-time>" << endl;
	cout << "  <mode>: bsp, tap, ssp:<n>, fsp, aap\n"
		<< "  <alg>: algorithm name. Support: lr, mlp, cnn, rnn, tm.\n"
		<< "  <param>: parameter of the algorithm, usually the shape of the algorithm.\n"
		<< "  <id-skip>: a list separated with space or comma.\n"
		<< "  [flex-param]: supports: interval:x (x is in seconds), portion:x (x in 0~1), "
		"improve:x,t (x: avg. imporovement, t: max waiting time), balance:w (num. of windows)\n"
		<< "  [mcast-param]: supports: all, ring:k, random:k,seed, hash:k\n"
		<< endl;
}

bool Option::preprocessMode(){
	for(char& ch : mode){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(mode, ":");
	vector<string> supported = { "bsp", "tap", "ssp", "fsp", "aap" };
	auto it = find(supported.begin(), supported.end(), t[0]);
	if(t[0] == "ssp"){
		if(t.size() > 1)
			sspGap = stoi(t[1]);
		else
			sspGap = 1;
	}
	return it != supported.end();
}

bool Option::processAlgorithm(){
	for(char& ch : algorighm){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> supported = { "lr", "mlp", "cnn", "rnn", "tm" };
	auto it = find(supported.begin(), supported.end(), algorighm);
	return it != supported.end();
}
