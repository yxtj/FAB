#include "TrainerFactory.h"
#include "GD.h"
#include "EM.h"

Trainer * TrainerFactory::generate(
	const std::string& name, const std::vector<std::string>& param)
{
	Trainer* p = nullptr;
	if(name == "gd"){
		p = new GD();
	} else if(name == "em"){
		p = new EM();
	}
	if(p != nullptr)
		p->init(param);
	return p;
}
