#pragma once
#include <utility>
#include <string>
#include <vector>
#include <initializer_list>

std::pair<int, int> getScreenSize();
bool beTrueOption(const std::string& str);

std::vector<int> getIntList(const std::string & str, const std::string& sepper = " ,");
std::vector<double> getDoubleList(const std::string & str, const std::string& sepper = " ,");
std::vector<std::string> getStringList(const std::string & str, const std::string& sepper = " ,");

// support range expressio i.e. 1,4-7,10 means 1,4,5,6,7,10
std::vector<int> getIntListByRange(const std::string & str, const std::string& sepper = " ,");

// binary: whether to use 2^10=1024 or 1000
int stoiKMG(const std::string & str, const bool binary = false);
size_t stoulKMG(const std::string & str, const bool binary = false);

bool contains(const std::string& str, const std::initializer_list<std::string>& list);
