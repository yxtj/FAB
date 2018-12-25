#include "FileEnumerator.h"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;

std::vector<std::string> FileEnumerator::listAll(
	const std::string& folder, const std::string& prefix)
{
	std::vector<std::string> res;
	path p(folder);
	if(!exists(p) || !is_directory(p))
		return res;
	directory_iterator end;
	for(auto it = directory_iterator(p); it != end; ++it){
		string fn = it->path().filename().string();
		if(prefix.empty() || prefix == fn.substr(0, prefix.size()))
			res.push_back(move(fn));
	}
	return res;
}

std::vector<std::string> FileEnumerator::listFile(
	const std::string& folder, const std::string& prefix)
{
	std::vector<std::string> res;
	path p(folder);
	if(!exists(p) || !is_directory(p))
		return res;
	directory_iterator end;
	for(auto it = directory_iterator(p); it != end; ++it){
		if(!is_regular_file(*it))
			continue;
		string fn = it->path().filename().string();
		if(prefix.empty() || prefix == fn.substr(0, prefix.size()))
			res.push_back(move(fn));
	}
	return res;
}

std::vector<std::string> FileEnumerator::listDirectory(
	const std::string& folder, const std::string& prefix)
{
	std::vector<std::string> res;
	path p(folder);
	if(!exists(p) || !is_directory(p))
		return res;
	directory_iterator end;
	for(auto it = directory_iterator(p); it != end; ++it){
		if(!is_directory(*it))
			continue;
		string fn = it->path().filename().string();
		if(prefix.empty() || prefix == fn.substr(0, prefix.size()))
			res.push_back(move(fn));
	}
	return res;
}

bool FileEnumerator::ensureDirectory(const std::string& folder){
	path p(folder);
	if(!exists(p)){
		return create_directories(p);
	}else if(!is_directory(p)){
		return false;
	}
	return true;
}
