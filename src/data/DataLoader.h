#pragma once
#include "DataHolder.h"
#include <string>
#include <vector>

class DataLoader{
	std::string ds_type;
	size_t npart, lid;
	bool localOnly;
public:
	void init(const std::string& dataset, const size_t nparts = 1,
		const size_t localid = 0, const bool localOnly = true);

	DataHolder load(const std::string& path, const bool trainPart,
		const std::vector<int> skips, const std::vector<int>& yIds, const bool header,
		const bool randomShuffle = false, const size_t topk = 0);
	
	void shuffle(DataHolder& dh);

private:
	void load_csv(DataHolder& dh, const std::string & fpath,
		const std::vector<int> skips, const std::vector<int>& yIds,
		const bool header, const size_t topk);
	void load_mnist(DataHolder& dh, const bool trainPart,
		const std::string & dpath, const size_t topk);
	void load_cifar10(DataHolder& dh, const bool trainPart, 
		const std::string & dpath, const size_t topk);
	void load_cifar100(DataHolder& dh, const bool trainPart, 
		const std::string & dpath, const size_t topk);
};
