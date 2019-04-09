#include "TrainerFactory.h"
#include "GD.h"
#include "EM.h"
#include "EM_KMeans.h"
#include "PSGD.h"
#include "BlockPSGD.h"
#include "psgd_poc/PSGD_point.h"
#include "psgd_poc/PSGD_dim.h"

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
	} else if(name == "bpsgd"){
		p = new BlockPSGD();
	} else if(name == "psgd_poc_point"){
		p = new PSGD_point();
	} else if(name == "psgd_poc_dim"){
		p = new PSGD_dim();
	}
	if(p != nullptr)
		p->init(param);
	return p;
}
