#pragma once
#include <string>
#include <vector>

struct ConfData {
	std::string dataset;
	std::string fnData;
	size_t topk;
	bool trainPart;

	std::vector<int> idSkip;
	std::vector<int> idY;
	std::string sepper;
	bool header;
	int lenUnit;

	bool normalize;
	bool shuffle;

	std::string fnOutput;
	bool binary;
	bool resume;

	size_t nw; // number of workers

	std::string mode;
	int staleGap; // the max gap between current processing iteration and the parameter iteratoin
	bool aapWait; // force fab wait for its gradient reply before continue
	int papOnlineProbeVersion;
	int papDynamicBatchSize; // search for the optimal global mini batch size online. the value is the method vesion
	bool papDynamicReportFreq; // search for the optimal report frequency (local micro batch size) online
	std::vector<std::string> intervalParam; // parameters for the flexible coordinator
	std::vector<std::string> mcastParam; // parameters for the multicast

	bool probe; // probe the hyper-parameters
	double probeRatio; // the ratio of data points used for each probe
	double probeMinGBSR; // the minimum global batch ratio used in probing
	//bool probeOnlineLoss; // use accumulated online loss or calculate loss with last model parameter
	bool probeLossFull; // calcualte the loss with full batch or the probed part

	std::string algorighm;
	std::string algParam;

	bool adjustSpeedRandom;
	bool adjustSpeedHetero;
	double speedRandomMin; // the minimum of adjustment factor
	double speedRandomMax; // the maximum of adjustment factor (larger values are reset to 0)
	std::vector<std::string> speedRandomParam;
	// one list per worker. pair<speed, time> means the worker is <speed> slower before <time>
	std::vector<std::vector<std::pair<double, double>>> speedHeterogenerity;

	std::string optimizer;
	std::vector<std::string> optimizerParam;
	//double lrate; // learning rate
	size_t batchSize;
	size_t reportSize;

	unsigned seed;

	// condition for archive result and write log
	size_t arvIter; // every n iterations
	double arvTime; // every t seconds
	int logIter;

	// termination condition
	size_t tcPoint; // maximum total used data point
	size_t tcDelta; // maximum total received delta/update report
	size_t tcIter; // maximum iteration
	double tcTime; // maximum training time
	double tcDiff; // minimum improvement cross iterations
};
