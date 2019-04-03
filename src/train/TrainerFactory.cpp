#include "TrainerFactory.h"
#include "GD.h"
#include "EM.h"
#include "EM_KMeans.h"
#include "PrioritizedSGD.h"
#include "PSGD_poc.h"

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
		p = new PrioritizedSGD();
	} else if(name == "psgd_poc"){
		p = new PSGD_poc();
	}
	if(p != nullptr)
		p->init(param);
	return p;
}
