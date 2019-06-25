#include "DataPoint.h"
#include <iostream>
#include <string>

using namespace std;

DataPoint parseLine(const std::string& line, const std::string& sepper,
	const std::unordered_set<int>& xIds, const std::unordered_set<int>& yIds)
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
	} catch(...){
		cout << "Error on idx= " << idx << " on line: " << line << endl;
	}
	return DataPoint{ {x}, y };
}

DataPoint parseLineVarLen(const std::string& line, const std::string& sepper,
	const int lenUnit, const std::unordered_set<int>& yIds)
{
	vector<vector<double>> x;
	vector<double> y;
	y.reserve(yIds.size());
	size_t p = line.find(sepper);
	size_t pl = 0;
	int idx = 0;
	try{
		vector<double> temp;
		while(p != string::npos){
			if(yIds.count(idx) != 0)
				y.push_back(stod(line.substr(pl, p - pl)));
			else{
				temp.push_back(stod(line.substr(pl, p - pl)));
				if(temp.size() == lenUnit){
					x.push_back(move(temp));
					temp.clear();
				}
			}
			pl = p + 1;
			p = line.find(sepper, pl);
			++idx;
		}
		if(yIds.count(idx) != 0)
			y.push_back(stod(line.substr(pl, p - pl)));
		else{
			temp.push_back(stod(line.substr(pl, p - pl)));
			if(temp.size() != lenUnit){
				throw invalid_argument("length of x does not match");
			}
			x.push_back(move(temp));
		}
	} catch(...){
		cout << "Error on idx= " << idx << " on line: " << line << endl;
	}
	return DataPoint{ x, y };
}

