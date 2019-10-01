#pragma once
#include <fstream>
#include <string>
#include "Parameter.h"

struct ParamArchiver {
	std::string fname;
	bool binary;
	bool resume;
	size_t wlen;
	std::fstream fs;
public:
	bool valid() const;
	bool init_write(const std::string& fname, const size_t wlen,
		const bool binary = false, const bool resume = false);
	bool init_read(const std::string& fname, const size_t wlen,
		const bool binary = false);
	void dump(const int iter, const double time, const size_t cnt, const Parameter& p);
	bool load(int& iter, double& time, size_t& cnt, Parameter& p);

	bool eof() const;
	void close();

	// only for dump and must be initialized with resume option
	bool load_nth(const int n, int& iter, double& time, size_t& cnt, Parameter& p);
	bool load_last(int& iter, double& time, size_t& cnt, Parameter& p);

private:
	void dump_text(const int iter, const double time, const size_t cnt, const Parameter& p);
	void dump_binary(const int iter, const double time, const size_t cnt, const Parameter& p);
	bool load_text(int& iter, double& time, size_t& cnt, Parameter& p);
	bool load_binary(int& iter, double& time, size_t& cnt, Parameter& p);
	bool load_last_text(int& iter, double& time, size_t& cnt, Parameter& p);
	bool load_last_binary(int& iter, double& time, size_t& cnt, Parameter& p);

	void parse_line(const std::string& line, int& iter, double& time, size_t& cnt, Parameter& param);
private:
	using fp_dump_t = void (ParamArchiver::*)(const int, const double, const size_t, const Parameter&);
	fp_dump_t pfd;
	using fp_load_t = bool (ParamArchiver::*)(int&, double&, size_t & cnt, Parameter&);
	fp_load_t pfl;
	size_t binWeightLen;
	size_t binUnitLen;
};
