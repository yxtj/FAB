#pragma once
#include "DataHolder.h"
#include <string>
#include <vector>

class DataLoader{
	std::string ds_type;
	size_t npart, pid;
	bool localOnly;

	// parameters for customized format (and csv, tsv)
	std::string sepper;
	int lunit = 0;
	std::vector<int> skips, yIds;
	bool header = false;
public:
	void init(const std::string& dataset, const size_t nparts = 1,
		const size_t localid = 0, const bool localOnly = true);

	void bindParameterTable(const std::string& sepper,
		const std::vector<int> skips, const std::vector<int>& yIds, const bool header);
	void bindParameterVarLen(const int lenUnit);

	DataHolder load(const std::string& path, const bool trainPart, const size_t topk = 0);

private:
	void load_customized(DataHolder& dh, const std::string & fpath,
		const std::string& sepper, const std::vector<int> skips, const std::vector<int>& yIds,
		const bool header, const size_t topk);
	void load_mnist(DataHolder& dh, const bool trainPart,
		const std::string & dpath, const size_t topk);
	void load_cifar10(DataHolder& dh, const bool trainPart, 
		const std::string & dpath, const size_t topk);
	void load_cifar100(DataHolder& dh, const bool trainPart, 
		const std::string & dpath, const size_t topk);
};
