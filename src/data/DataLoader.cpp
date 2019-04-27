#include "DataLoader.h"
#include <fstream>
using namespace std;

// -------- DataLoader basic --------

void DataLoader::init(const std::string & dataset, const size_t nparts,
	const size_t localid, const bool onlyLocalPart)
{
	ds_type = dataset;
	npart = nparts;
	lid = localid;
	localOnly = onlyLocalPart;
}

DataHolder DataLoader::load(const std::string & path, const bool trainPart,
	const std::vector<int> skips, const std::vector<int>& yIds,
	const bool header, const bool randomShuffle, const size_t topk)
{
	DataHolder dh(npart, lid);
	if(ds_type == "csv"){
		load_csv(dh, path, skips, yIds, header, topk);
	} else if(ds_type == "mnist"){
		load_mnist(dh, trainPart, path, topk);
	} else if(ds_type == "cifar10"){
		load_cifar10(dh, trainPart, path, topk);
	} else if(ds_type == "cifar100"){
		load_cifar100(dh, trainPart, path, topk);
	}
	if(randomShuffle)
		shuffle(dh);
	return dh;
}

void DataLoader::shuffle(DataHolder & dh)
{
	dh.shuffle();
}

// -------- load csv file --------
void DataLoader::load_csv(DataHolder & dh, const std::string & fpath,
	const std::vector<int> skips, const std::vector<int>& yIds, 
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
	size_t p = line.find(",");
	while(p != string::npos){
		++n;
		p = line.find(",", p + 1);
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

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t lid = 0; // line id;
	while(getline(fin, line)){
		if(line.size() < nx || line.front() == '#') // invalid line
			continue;
		if(localOnly && lid++ % npart != lid) // not local line
			continue;
		if(topk != 0 && lid > topk)
			break;
		DataPoint dp = parseLine(line, ",", xIds, yIds_u);
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
	ifstream fimg(dpath + '/' + prefix + "-images-idx3-ubyte");
	if(fimg.fail()){
		throw invalid_argument("Error in reading file: " + dpath + '/' + prefix + "-images-idx3-ubyte");
	}
	ifstream flbl(dpath + '/' + prefix + "-labels-idx1-ubyte");
	if(flbl.fail()){
		throw invalid_argument("Error in reading file: " + dpath + '/' + prefix + "-labels-idx3-ubyte");
	}
	constexpr int len = 28 * 28;
	char buffer[len];
	fimg.read(buffer, 16);
	flbl.read(buffer, 8);
	for(size_t i = 0; i < n; ++i){
		fimg.read(buffer, len);
		int lbl = flbl.get();
		vector<double> x(len), y(1);
		for(int j = 0; j < len; ++j)
			x[j] = static_cast<double>(buffer[j]);
		y[0] = static_cast<double>(lbl);
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
		constexpr int len = 3 * 32 * 32;
		char buffer[len+1];
		for(size_t i = 0; i < n; ++i){
			fin.read(buffer, len+1);
			vector<double> x(len), y(1);
			y[0] = static_cast<double>(buffer[0]);
			for(int j = 0; j < len; ++j)
				x[j] = static_cast<double>(buffer[j + 1]);
			dh.add(move(x), move(y));
		}
	};
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
		for(size_t i = 0; i < n; ++i){
			fin.read(buffer, len + 2);
			vector<double> x(len), y(1);
			y[0] = static_cast<double>(buffer[1]);
			for(int j = 0; j < len; ++j)
				x[j] = static_cast<double>(buffer[j + 2]);
			dh.add(move(x), move(y));
		}
	};
	if(trainPart){
		size_t n = topk == 0 ? 50000 : topk;
		func(dpath + "/train.bin", n);
	} else{
		size_t n = topk == 0 ? 10000 : topk;
		func(dpath + "/test.bin", n);
	}

}

