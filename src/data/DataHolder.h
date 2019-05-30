#pragma once
#include "DataPoint.h"
#include <vector>
#include <string>

class DataHolder {
	std::vector<DataPoint> data;
	size_t npart; // total number of parts
	size_t pid; // part id

	size_t nx; // length of x
	size_t ny; // length of y
public:
	// <nparts> <localid>: used for distributed case
	DataHolder(const size_t nparts = 1, const size_t localid = 0);
	void setLength(const size_t lx, const size_t ly);
	size_t xlength() const;
	size_t ylength() const;
	size_t nparts() const;
	size_t partid() const;

	// give the column id of y and the skipped ones, the rest are x. id starts from 0
	// throw exceptions if something wrong
	void load(const std::string& fpath, const std::string& sepper,
		const std::vector<int> skips, const std::vector<int>& yIds,
		const bool header, const bool onlyLocalPart = false, const size_t topk = 0);

	void add(const std::vector<double>& x, const std::vector<double>& y);
	void add(std::vector<double>&& x, std::vector<double>&& y);
	void add(const DataPoint& dp);
	void add(DataPoint&& dp);
	
	void shuffle();

	size_t size() const {
		return data.size();
	}
	const DataPoint& get(const size_t idx) const {
		return data[idx];
	}

	// normalize to [-1, 1]
	void normalize(const bool onY);
};
