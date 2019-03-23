#pragma once
#include <fstream>
#include <string>
#include "Parameter.h"

struct ParamArchiver {
	std::string fname;
	bool binary;
	size_t wlen;
	std::fstream fs;
public:
	bool valid() const;
	bool init_write(const std::string& fname, const size_t wlen,
		const bool append = false, const bool binary = false);
	bool init_read(const std::string& fname, const size_t wlen,
		const bool binary = false);
	void dump(const int iter, const double time, const Parameter& p);
	bool load(int& iter, double& time, Parameter& p);
	void close();

	bool load_last(int& iter, double& time, Parameter& p);

private:
	void dump_text(const int iter, const double time, const Parameter& p);
	void dump_binary(const int iter, const double time, const Parameter& p);
	bool load_text(int& iter, double& time, Parameter& p);
	bool load_binary(int& iter, double& time, Parameter& p);
	bool load_last_text(int& iter, double& time, Parameter& p);
	bool load_last_binary(int& iter, double& time, Parameter& p);

	void parse_line(const std::string& line, int& iter, double& time, Parameter& param);
private:
	using fp_dump_t = void (ParamArchiver::*)(const int, const double, const Parameter&);
	fp_dump_t pfd;
	using fp_load_t = bool (ParamArchiver::*)(int&, double&, Parameter&);
	fp_load_t pfl;
	size_t binUnitLen;
};
