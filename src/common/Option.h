#pragma once
#include <string>
#include <vector>

struct Option{
	//std::string prefix;
	std::string fnData;
	std::string fnOutput;
	std::vector<int> idSkip;
	std::vector<int> idY;
	bool header;
	bool normalize;
	bool binary;

	size_t nw; // number of workers

	std::string mode;
	int staleGap; // the max gap between current processing iteration and the parameter iteratoin
	bool aapWait; // force fab wait for its gradient reply before continue
	std::vector<std::string> intervalParam; // parameters for the flexible coordinator
	std::vector<std::string> mcastParam; // parameters for the multicast

	std::string algorighm;
	std::string algParam;

	std::string optimizer;
	std::vector<std::string> optimizerParam;
	//double lrate; // learning rate
	size_t batchSize;

	unsigned seed;

	// condition for archive result and write log
	size_t arvIter; // every n iterations
	double arvTime; // every t seconds
	int logIter;
	
	// termination condition
	size_t tcIter; // maximum iteration
	double tcTime; // maximum training time
	double tcDiff; // minimum improvement cross iterations

	bool parse(int argc, char* argv[], const size_t nWorker);
	void showUsage() const;

private:
	bool preprocessMode();
	bool processAlgorithm();
	bool processOptimizer();

	struct Impl;
	Impl* pimpl = nullptr;
};
