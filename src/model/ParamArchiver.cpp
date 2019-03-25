#include "ParamArchiver.h"

using namespace std;

bool ParamArchiver::valid() const
{
	return !fname.empty() && fs.is_open() && fs.good();
}

bool ParamArchiver::init_write(const std::string & fname,
	const size_t wlen, const bool append, const bool binary)
{
	this->fname = fname;
	this->wlen = wlen;
	this->binary = binary;
	ios_base::openmode mode = ios_base::out;
	if(append)
		mode |= ios_base::ate;
	if(binary){
		mode |= ios_base::binary;
		pfd = &ParamArchiver::dump_binary;
		binWeightLen = wlen * sizeof(double);
		binUnitLen = sizeof(int) + sizeof(double) + binWeightLen;
	} else{
		pfd = &ParamArchiver::dump_text;
	}
	fs = fstream(fname, mode);
	return fs.is_open();
}

bool ParamArchiver::init_read(const std::string & fname,
	const size_t wlen, const bool binary)
{
	this->fname = fname;
	this->wlen = wlen;
	this->binary = binary;
	ios_base::openmode mode = ios_base::in;
	if(binary){
		mode |= ios_base::binary;
		pfl = &ParamArchiver::load_binary;
		binWeightLen = wlen * sizeof(double);
		binUnitLen = sizeof(int) + sizeof(double) + binWeightLen;
	} else{
		pfl = &ParamArchiver::load_text;
	}
	fs = fstream(fname, mode);
	return fs.is_open();
}

void ParamArchiver::dump(const int iter, const double time, const Parameter & p)
{
	(this->*pfd)(iter, time, p);
}

bool ParamArchiver::load(int & iter, double & time, Parameter & p)
{
	return (this->*pfl)(iter, time, p);
}

bool ParamArchiver::eof() const
{
	return fs.eof();
}

void ParamArchiver::close()
{
	fs.close();
	fs.clear();
}

bool ParamArchiver::load_last(int & iter, double & time, Parameter & p)
{
	if(binary){
		return load_last_binary(iter, time, p);
	} else{
		return load_last_text(iter, time, p);
	}
}

// private implementation

void ParamArchiver::dump_text(const int iter, const double time, const Parameter & p)
{
	fs << iter << "," << time;
	//assert(wlen == p.weights);
	for(auto& v : p.weights){
		fs << "," << v;
	}
	fs << "\n";
}

void ParamArchiver::dump_binary(const int iter, const double time, const Parameter & p)
{
	fs.write(reinterpret_cast<const char*>(&iter), sizeof(iter));
	fs.write(reinterpret_cast<const char*>(&time), sizeof(time));
	fs.write(reinterpret_cast<const char*>(p.weights.data()), binWeightLen);
}

bool ParamArchiver::load_text(int & iter, double & time, Parameter & p)
{
	string line;
	getline(fs, line);
	if(fs.fail() || line.size() < 4)
		return false;
	parse_line(line, iter, time, p);
	return p.weights.size() == wlen;
}

bool ParamArchiver::load_binary(int & iter, double & time, Parameter & p)
{
	fs.read(reinterpret_cast<char*>(&iter), sizeof(iter));
	fs.read(reinterpret_cast<char*>(&time), sizeof(time));
	vector<double> weights(wlen);
	fs.read(reinterpret_cast<char*>(weights.data()), binWeightLen);
	p.weights = move(weights);
	p.n = p.weights.size();
	return bool(fs);
}

bool ParamArchiver::load_last_text(int & iter, double & time, Parameter & p)
{
	fs.seekg(-(20 + 30 * wlen), ios_base::end); // no more than 29 characters per number
	string line, lastline;
	getline(fs, lastline);
	while(getline(fs, line)){
		if(line.size() <= 3)
			continue;
		lastline = move(line);
	}
	if(fs.tellg() == 0 || lastline.size() <= 3)
		return false;
	parse_line(lastline, iter, time, p);
	return p.weights.size() == wlen;
}

bool ParamArchiver::load_last_binary(int & iter, double & time, Parameter & p)
{
	fs.seekg(-binUnitLen, ios_base::end);
	if(fs.tellg() == 0)
		return false;
	load_binary(iter, time, p);
	return false;
}

void ParamArchiver::parse_line(const std::string& line, int & iter, double & time, Parameter & param)
{
	size_t pl = 0;
	size_t p = line.find(',');
	iter = stoi(line.substr(pl, p - pl)); // iteration-number
	pl = p + 1;
	p = line.find(',', pl);
	time = stod(line.substr(pl, p - pl)); // time
	pl = p + 1;
	p = line.find(',', pl);
	vector<double> weights;
	// weights
	while(p != string::npos){
		weights.push_back(stod(line.substr(pl, p - pl)));
		pl = p + 1;
		p = line.find(',', pl);
	}
	weights.push_back(stod(line.substr(pl)));
	param.weights = move(weights);
}
