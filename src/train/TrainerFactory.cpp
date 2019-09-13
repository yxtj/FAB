#include "TrainerFactory.h"
#include "GD.h"
#include "EM.h"
#include "EM_KMeans.h"
#include "PSGD.h"
#include "psgd_poc/PSGDBlock.h"
#include "psgd_poc/PSGDDecay.h"
#include "psgd_poc/PSGD_point.h"
#include "psgd_poc/PSGD_dim.h"
#include <algorithm>

using namespace std;

std::vector<std::string> TrainerFactory::supportList()
{
	static vector<string> supported = { "gd", "em", "kmeans", "psgd", "psgdb", "psgdd" };
	return supported;
}

bool TrainerFactory::isSupported(const std::string& name)
{
	const auto&& supported = supportList();
	auto it = find(supported.begin(), supported.end(), name);
	if(it == supported.end() && name.find("_poc_") == string::npos)
		return false;
	return true;
}


Trainer * TrainerFactory::generate(
	const std::string& name, const std::vector<std::string>& param)
{
	Trainer* p = nullptr;
	if(name == "gd"){
		p = new GD();
	} else if(name == "em"){
		p = new EM();
	} else if(name == "kmeans"){
		p = new EM_KMeans();
	} else if(name == "psgd"){
		p = new PSGD();
	} else if(name == "psgdb"){
		p = new PSGDBlock();
	} else if(name == "psgdd"){
		p = new PSGDDecay();
	} else if(name == "psgd_poc_point"){
		p = new PSGD_point();
	} else if(name == "psgd_poc_dim"){
		p = new PSGD_dim();
	}
	if(p != nullptr)
		p->init(param);
	return p;
}
