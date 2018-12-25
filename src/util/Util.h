#pragma once
#include <utility>
#include <string>
#include <vector>

std::pair<int, int> getScreenSize();
bool beTrueOption(const std::string& str);

std::vector<int> getIntList(const std::string & str, const std::string& sepper = " ,");
std::vector<double> getDoubleList(const std::string & str, const std::string& sepper = " ,");

// binary: whether to use 2^10=1024 or 1000
int stoiKMG(const std::string & str, const bool binary = false);
size_t stoulKMG(const std::string & str, const bool binary = false);
