#pragma once
#include <unordered_map>
#include <vector>

// map network-id with logic-id
struct IDMapper {
	void registerID(const int nid, const int lid);
	// return <whether-registered, registered-logic-id/net-id>
	std::pair<bool, int> nidTrans(const int nid) const;
	int nid2lid(const int nid) const;
	int lid2nid(const int lid) const;
	// list all nid-lid pairs
	std::vector<std::pair<int, int>> list() const;

private:
	std::unordered_map<int, int> contN2L;
	std::unordered_map<int, int> contL2N;
};
