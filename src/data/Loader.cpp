#include "Loader.h"
#include <iostream>

using namespace std;

DataPoint parseLine(const std::string& line, const std::string& sepper,
	const std::unordered_set<int>& xIds, const std::unordered_set<int>& yIds, const bool appOne)
{
	vector<double> x;
	x.reserve(xIds.size());
	vector<double> y;
	y.reserve(yIds.size());
	size_t p = line.find(sepper);
	size_t pl = 0;
	int idx = 0;
	try{
		while(p != string::npos){
			if(xIds.count(idx) != 0)
				x.push_back(stod(line.substr(pl, p - pl))); 
			else if(yIds.count(idx) != 0)
				y.push_back(stod(line.substr(pl, p - pl)));
			pl = p + 1;
			p = line.find(sepper, pl);
			++idx;
		}
		if(xIds.count(idx) != 0)
			x.push_back(stod(line.substr(pl, p - pl)));
		else if(yIds.count(idx) != 0)
			y.push_back(stod(line.substr(pl, p - pl)));
		if(appOne)
			x.push_back(1.0);
	} catch(...){
		cout << "Error on idx= " << idx << " on line: " << line << endl;
	}
	return DataPoint{x, y};
}

