#pragma once
#include <string>
#include <vector>
#include <fstream>

struct PriorityDumper {
	bool init(const std::string& fname, bool obinary, bool ofloat, bool resume = false, std::streampos pos = 0);
	void dump(const std::vector<double>& priority)
	{
		(this->*pfd)(priority);
	}
	void dump(const std::vector<float>& priority)
	{
		(this->*pff)(priority);
	}
private:
	std::ofstream fout;
	using pfd_t = void (PriorityDumper::*)(const std::vector<double>&);
	pfd_t pfd;
	using pff_t = void (PriorityDumper::*)(const std::vector<float>&);
	pff_t pff;
	void dumpCSV_d(const std::vector<double>& priority);
	void dumpCSV_f(const std::vector<float>& priority);
	void dumpDouble_d(const std::vector<double>& priority);
	void dumpDouble_f(const std::vector<float>& priority);
	void dumpFloat_d(const std::vector<double>& priority);
	void dumpFloat_f(const std::vector<float>& priority);
};
