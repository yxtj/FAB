#include "DataLoader.h"
#include <fstream>
#include <limits>
#include <algorithm>
using namespace std;

// -------- DataLoader basic --------

std::vector<std::string> DataLoader::supportList()
{
	static vector<string> supported{ "customize", "csv", "tsv", "list",
		"mnist", "cifar10", "cifar100" };
	return supported;
}

bool DataLoader::isSupported(const std::string& name)
{
	const auto&& supported = supportList();
	auto it = find(supported.begin(), supported.end(), name);
	return it != supported.end();
}

void DataLoader::init(const std::string & dataset, const size_t nparts,
	const size_t localid, const bool onlyLocalPart)
{
	ds_type = dataset;
	npart = nparts;
	pid = localid;
	localOnly = onlyLocalPart;
}

void DataLoader::bindParameterTable(const std::string & sepper,
	const std::vector<int> skips, const std::vector<int>& yIds, const bool header)
{
	this->sepper = sepper;
	this->skips = skips;
	this->yIds = yIds;
	this->header = header;
}

void DataLoader::bindParameterVarLen(const std::string& sepper, const int lenUnit, const std::vector<int>& yIds){
	this->sepper = sepper;
	this->lunit = lenUnit;
	this->yIds = yIds;
}

DataHolder DataLoader::load(
	const std::string & path, const bool trainPart, const size_t topk)
{
	DataHolder dh(npart, pid);
	size_t limit = topk == 0 ? numeric_limits<size_t>::max() : topk;
	if(ds_type == "csv"){
		load_customized(dh, path, ",", skips, yIds, header, limit);
	} else if(ds_type == "tsv"){
		load_customized(dh, path, "\t", skips, yIds, header, limit);
	} else if(ds_type == "customize"){
		load_customized(dh, path, sepper, skips, yIds, header, limit);
	} else if(ds_type == "list"){
		load_varlist(dh, path, lunit, sepper, yIds, limit);
	} else if(ds_type == "mnist"){
		load_mnist(dh, trainPart, path, limit);
	} else if(ds_type == "cifar10"){
		load_cifar10(dh, trainPart, path, limit);
	} else if(ds_type == "cifar100"){
		load_cifar100(dh, trainPart, path, limit);
	}
	return dh;
}

// -------- load customized file --------
void DataLoader::load_customized(DataHolder & dh, const std::string & fpath,
	const std::string& sepper, const std::vector<int> skips, const std::vector<int>& yIds,
	const bool header, const size_t topk)
{
	ifstream fin(fpath);
	if(fin.fail()){
		throw invalid_argument("Error in reading file: " + fpath);
	}
	// calculate number of x
	size_t nx = 0;
	int n = 0;
	string line;
	getline(fin, line);
	size_t p = line.find(sepper);
	while(p != string::npos){
		++n;
		p = line.find(sepper, p + 1);
	}
	++n;
	// calculate xIds and yId
	unordered_set<int> xIds;
	unordered_set<int> yIds_u;
	for(int i = 0; i < n; ++i){
		if(find(skips.begin(), skips.end(), i) != skips.end())
			continue;
		if(find(yIds.begin(), yIds.end(), i) != yIds.end())
			yIds_u.insert(i);
		else
			xIds.insert(i);
	}
	//nx = n - skips.size() - 1; // does not work when skips contains placeholders like -1
	nx = xIds.size();
	size_t ny = yIds_u.size();
	dh.setLength(nx, ny);

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t i = 0; // line id;
	while(getline(fin, line)){
		if(line.size() < nx || line.front() == '#') // invalid line
			continue;
		if(localOnly && i++ % npart != pid) // not local line
			continue;
		if(topk != 0 && i > topk)
			break;
		DataPoint dp = parseLine(line, sepper, xIds, yIds_u);
		dh.add(move(dp));
	}
}

void DataLoader::load_varlist(DataHolder& dh, const std::string & fpath, const int lunit,
	const std::string& sepper, const std::vector<int>& yIds, const size_t topk)
{
	ifstream fin(fpath);
	if(fin.fail()){
		throw invalid_argument("Error in reading file: " + fpath);
	}
	// calculate number of x
	size_t nx = 0;
	int n = 0;
	string line;
	getline(fin, line);
	size_t p = line.find(sepper);
	while(p != string::npos){
		++n;
		p = line.find(sepper, p + 1);
	}
	++n;
	// calculate yId
	unordered_set<int> yIds_u;
	for(int i = 0; i < n; ++i){
		if(find(yIds.begin(), yIds.end(), i) != yIds.end())
			yIds_u.insert(i);
	}
	//nx = n - skips.size() - 1; // does not work when skips contains placeholders like -1
	size_t ny = yIds_u.size();
	dh.setLength(static_cast<size_t>(lunit), ny);

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t i = 0; // line id;
	while(getline(fin, line)){
		if(line.size() < nx || line.front() == '#') // invalid line
			continue;
		if(localOnly && i++ % npart != pid) // not local line
			continue;
		if(topk != 0 && i > topk)
			break;
		DataPoint dp = parseLineVarLen(line, sepper, lunit, yIds_u);
		dh.add(move(dp));
	}
}

// -------- load MNIST --------
void DataLoader::load_mnist(DataHolder & dh, const bool trainPart,
	const std::string & dpath, const size_t topk)
{
	string prefix = trainPart ? "train" : "t10k";
	size_t n = trainPart ? 60000 : 10000;
	n = n < topk ? n : topk;
	ifstream fimg(dpath + '/' + prefix + "-images.idx3-ubyte");
	if(fimg.fail()){
		throw invalid_argument("Error in reading file: " + dpath + '/' + prefix + "-images.idx3-ubyte");
	}
	ifstream flbl(dpath + '/' + prefix + "-labels.idx1-ubyte");
	if(flbl.fail()){
		throw invalid_argument("Error in reading file: " + dpath + '/' + prefix + "-labels.idx3-ubyte");
	}
	constexpr int len = 28 * 28;
	dh.setLength(len, 10);
	char buffer[len];
	fimg.read(buffer, 16);
	flbl.read(buffer, 8);
	for(size_t i = 0; i < n && fimg && flbl; ++i){
		fimg.read(buffer, len);
		int lbl = flbl.get();
		if(localOnly && i % npart != pid) // not local line
			continue;
		vector<double> x(len), y(10, 0.0);
		for(int j = 0; j < len; ++j)
			x[j] = static_cast<double>(buffer[j]);
		y[lbl] = 1.0;
		dh.add(move(x), move(y));
	}
}

// -------- load CIFAR10 --------
void DataLoader::load_cifar10(DataHolder & dh, const bool trainPart,
	const std::string & dpath, const size_t topk)
{
	auto func = [&](const std::string& name, const size_t n){
		ifstream fin(name);
		if(fin.fail()){
			throw invalid_argument("Error in reading file: " + name);
		}
		constexpr int len = 3 * 32 * 32; // RGB - X - Y
		char buffer[len+1];
		for(size_t i = 0; i < n && fin; ++i){
			fin.read(buffer, len+1);
			if(localOnly && i % npart != pid) // not local line
				continue;
			vector<double> x(len), y(10, 0.0);
			const int p = static_cast<int>(buffer[0]);
			y[p] = 1.0;
			for(int j = 0; j < len; ++j)
				x[j] = static_cast<double>(buffer[j + 1]);
			dh.add(move(x), move(y));
		}
	};
	dh.setLength(3 * 32 * 32, 10);
	if(trainPart){
		size_t n = topk == 0 ? 50000 : topk;
		for(int fileId = 1; fileId <= 5; ++fileId){
			func(dpath + "/data_batch_" + to_string(fileId) + ".bin",
				(n <= 10000 ? n : 10000));
			if(n <= 10000)
				break;
			n -= 10000;
		}
	} else{
		size_t n = topk == 0 ? 10000 : topk;
		func(dpath + "/test_batch.bin", topk);
	}
}

// -------- load CIFAR100 --------
void DataLoader::load_cifar100(DataHolder & dh, const bool trainPart,
	const std::string & dpath, const size_t topk)
{
	auto func = [&](const std::string& name, const size_t n){
		ifstream fin(name);
		if(fin.fail()){
			throw invalid_argument("Error in reading file: " + name);
		}
		constexpr int len = 3 * 32 * 32; // RGB - X - Y
		char buffer[len + 2];
		for(size_t i = 0; i < n && fin; ++i){
			fin.read(buffer, len + 2);
			if(localOnly && i % npart != pid) // not local line
				continue;
			vector<double> x(len), y(100, 0.0);
			const int p = static_cast<int>(buffer[1]);
			y[p] = 1.0;
			for(int j = 0; j < len; ++j)
				x[j] = static_cast<double>(buffer[j + 2]);
			dh.add(move(x), move(y));
		}
	};
	dh.setLength(3 * 32 * 32, 100);
	if(trainPart){
		size_t n = topk == 0 ? 50000 : topk;
		func(dpath + "/train.bin", n);
	} else{
		size_t n = topk == 0 ? 10000 : topk;
		func(dpath + "/test.bin", n);
	}

}

