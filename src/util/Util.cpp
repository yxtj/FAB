#include "Util.h"
#include <vector>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define _OS_WIN
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

using namespace std;

std::pair<int, int> getScreenSize() {
	int cols = -1;
	int lines = -1;

#ifdef _OS_WIN
//	printf("win\n");
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	cols = csbi.dwSize.X;
	lines = csbi.dwSize.Y;
#else
//	printf("posix\n");
	struct winsize ts;
	ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
	cols = ts.ws_col;
	lines = ts.ws_row;
#endif 
	return make_pair(cols, lines);
}

#undef _OS_WIN

bool beTrueOption(const std::string& str){
	static vector<string> true_options({"1", "t", "T", "true", "True", "TRUE", "y", "Y", "yes", "Yes", "YES"});
	return find(true_options.begin(), true_options.end(), str) != true_options.end();
}


std::vector<int> getIntList(const std::string & str, const std::string& sepper)
{
	std::vector<int> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(stoi(str.substr(pl, p - pl)));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(stoi(str.substr(pl)));
	return res;
}

std::vector<double> getDoubleList(const std::string & str, const std::string& sepper)
{
	std::vector<double> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(stod(str.substr(pl, p - pl)));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(stod(str.substr(pl)));
	return res;
}

std::vector<std::string> getStringList(const std::string & str, const std::string& sepper)
{
	std::vector<std::string> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(str.substr(pl, p - pl));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(str.substr(pl));
	return res;
}

std::vector<int> getIntListByRange(const std::string & str, const std::string & sepper)
{
	vector<string> sl = getStringList(str, sepper);
	vector<int> res;
	for(auto& s : sl){
		if(s.empty())
			continue;
		auto p = s.find('-');
		if(p != 0 && p != string::npos){ // 2-10
			int f = stoi(s.substr(0, p));
			int l = stoi(s.substr(p + 1));
			while(f <= l){
				res.push_back(f);
				++f;
			}
		} else{ // -1 or 3
			res.push_back(stoi(s));
		}
	}
	return res;
}

int stoiKMG(const std::string & str, const bool binary)
{
	if(str.empty())
		return 0;
	const int f = binary ? 1024 : 1000;
	char ch = str.back();
	int factor = 1;
	if(ch == 'k' || ch == 'K')
		factor = f;
	else if(ch == 'm' || ch == 'M')
		factor = f * f;
	else if(ch == 'g' || ch == 'G')
		factor = f * f * f;
	return stoi(str)*factor;
}

size_t stoulKMG(const std::string & str, const bool binary)
{
	if(str.empty())
		return 0;
	const size_t f = binary ? 1024 : 1000;
	char ch = str.back();
	size_t factor = 1;
	if(ch == 'k' || ch == 'K')
		factor = f;
	else if(ch == 'm' || ch == 'M')
		factor = f * f;
	else if(ch == 'g' || ch == 'G')
		factor = f * f * f;
	return stoul(str)*factor;
}

bool contains(const std::string& str, const std::initializer_list<std::string>& list)
{
	return find(list.begin(), list.end(), str) != list.end();
}
