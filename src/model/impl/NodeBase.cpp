#include "NodeBase.h"
#include <numeric>

using namespace std;

// ---- NodeBase ----

NodeBase::NodeBase(const size_t offset, const std::vector<int>& shape)
	: off(offset), shape(shape)
{
	if(shape.empty()){
		nw = 0;
	}else{
		nw = 1;
		for(auto& v : shape)
			nw *= v;
	}
}

size_t NodeBase::nweight() const
{
	return nw;
}
