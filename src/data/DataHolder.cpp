#include "DataHolder.h"
#include <algorithm>
#include <unordered_set>
#include <fstream>

using namespace std;

DataHolder::DataHolder(const size_t nparts, const size_t localid, const bool varx)
	:  npart(nparts), pid(localid), varx(varx)
{}

void DataHolder::setLength(const size_t lx, const size_t ly){
	nx = lx;
	ny = ly;
}

size_t DataHolder::xlength() const{
	return nx;
}

size_t DataHolder::ylength() const{
	return ny;
}

size_t DataHolder::partid() const
{
	return pid;
}

size_t DataHolder::nparts() const
{
	return npart;
}

void DataHolder::load(const std::string& fpath, const std::string& sepper,
	const std::vector<int> skips, const std::vector<int>& yIds,
	const bool header, const bool onlyLocalPart, const size_t topk)
{
	ifstream fin(fpath);
	if(fin.fail()){
		throw invalid_argument("Error in reading file: " + fpath);
	}
	// calculate number of x
	nx = 0;
	int n = 0;
	string line;
	getline(fin, line);
	size_t p = line.find(sepper);
	while(p!=string::npos){
		++n;
		p = line.find(sepper, p+1);
	}
	++n;
	// calculate xIds and yId
	unordered_set<int> xIds;
	unordered_set<int> yIds_u;
	for(int i=0; i<n; ++i){
		if(find(skips.begin(), skips.end(), i) != skips.end())
			continue;
		if(find(yIds.begin(), yIds.end(), i) != yIds.end())
			yIds_u.insert(i);
		else
			xIds.insert(i);
	}
	//nx = n - skips.size() - 1; // does not work when skips contains placeholders like -1
	nx = xIds.size();
	ny = yIds_u.size();

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t lid = 0; // line id;
	while(getline(fin, line)){
		if(line.size() < nx || line.front() == '#') // invalid line
			continue;
		if(onlyLocalPart && lid++ % npart != pid) // not local line
			continue;
		if(topk != 0 && lid > topk)
			break;
		DataPoint dp = parseLine(line, sepper, xIds, yIds_u);
		data.push_back(move(dp));
	}
}

void DataHolder::add(const std::vector<double>& x, const std::vector<double>& y){
	DataPoint dp{ {x},y };
	data.push_back(move(dp));
}

void DataHolder::add(std::vector<double>&& x, std::vector<double>&& y){
	DataPoint dp;
	dp.x.resize(1);
	dp.x[0] = move(x);
	dp.y = move(y);
	data.push_back(move(dp));
}

void add(const std::vector<std::vector<double>>& x, const std::vector<double>& y){
	DataPoint dp{ x,y };
	data.push_back(move(dp));
}

void add(std::vector<std::vector<double>>&& x, std::vector<double>&& y){
	DataPoint dp{ move(x), move(y) };
	data.push_back(move(dp));
}

void DataHolder::add(const DataPoint & dp)
{
	data.push_back(dp);
}

void DataHolder::add(DataPoint && dp)
{
	data.push_back(move(dp));
}

void DataHolder::shuffle()
{
	random_shuffle(data.begin(), data.end());
}

// normalize to [-1, 1]
void DataHolder::normalize(const bool onY)
{
	if(data.size() < 2)
		return;
	vector<double> max_x = data.front().x;
	vector<double> min_x = data.front().x;
	vector<double> max_y = data.front().y;
	vector<double> min_y = data.front().y;
	for(const auto& d : data){
		for(auto& x : d.x){
			for(size_t i = 0; i < nx; ++i){
				if(x[i] > max_x[i])
					max_x[i] = x[i];
				else if(x[i] < min_x[i])
					min_x[i] = x[i];
			}
		}
		if(onY){
			for(size_t i = 0; i < ny; ++i){
				if(d.y[i] > max_y[i])
					max_y[i] = d.y[i];
				else if(d.y[i] < min_y[i])
					min_y[i] = d.y[i];
			}
		}
	}
	vector<double> range_x(nx);
	vector<double> range_y(ny);
	for(size_t i = 0; i < nx; ++i){
		range_x[i] = max_x[i] - min_x[i];
		if(range_x[i] == 0.0)
			range_x[i] = 1.0;
	}
	for(size_t i = 0; i < ny; ++i)
		range_y[i] = max_y[i] - min_y[i];
	for(auto&d : data){
		for(auto& x : data.x)
			for(size_t i = 0; i < nx; ++i)
				x[i] = 2 * (x[i] - min_x[i]) / range_x[i] - 1;
		if(onY){
			for(size_t i = 0; i < ny; ++i)
				d.y[i] = 2 * (d.y[i] - min_y[i]) / range_y[i] - 1;
		}
	}
}
