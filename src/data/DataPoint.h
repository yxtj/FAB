#pragma once
#include <vector>
#include <unordered_set>

struct DataPoint {
	std::vector<std::vector<double>> x;
	std::vector<double> y;
};

DataPoint parseLine(const std::string& line, const std::string& sepper,
	const std::unordered_set<int>& xIds, const std::unordered_set<int>& yIds);

DataPoint parseLineVarLen(const std::string& line, const std::string& sepper,
	const int lenUnit, const std::unordered_set<int>& yIds);
