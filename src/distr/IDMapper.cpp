#include "IDMapper.h"
using namespace std;

void IDMapper::registerID(const int nid, const int lid){
	contN2L[nid] = lid;
	contL2N[lid] = nid;
}

// return <whether-a-worker, worker/master-id>
std::pair<bool, int> IDMapper::nidTrans(const int nid) const {
	auto it = contN2L.find(nid);
	if(it == contN2L.end()){
		return make_pair(false, nid);
	} else{
		return make_pair(true, it->second);
	}
}

int IDMapper::nid2lid(const int nid) const {
	return contN2L.at(nid);
}
int IDMapper::lid2nid(const int lid) const {
	return contL2N.at(lid);
}

std::vector<std::pair<int, int>> IDMapper::list() const{
	std::vector<pair<int, int>> temp;
	for(auto& p : contN2L)
		temp.emplace_back(p.first, p.second);
	return temp;
}
