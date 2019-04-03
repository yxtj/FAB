#pragma once
#include <unordered_map>
#include <boost/heap/fibonacci_heap.hpp>
//#include <boost/heap/binomal_heap.hpp>

template <class id_t, class hash_t = std::hash<id_t>>
struct PrioritizedHolder{

	id_t top();
	void pop();

	void update(const id_t& k, const float& p);
	void reset(const id_t& k);

	size_t size() const { return heap.size(); }
	bool empty() const { return heap.empty(); }
	void clear() {
		heap.clear();
		khm.clear();
	}
	void reserve(const size_t n) {
		khm.reserve(n);
	}

private:
	struct Unit{
		id_t k;
		float p;
	};
	struct CmpUnit{
		bool operator()(const Unit& a, const Unit& b) const {
			return a.p < b.p;
		}
	};
	struct HashID{
		size_t operator()(const id_t& k) const{
			return (k.first << 5) | (k.second);
		}
	};
	using heap_t = boost::heap::fibonacci_heap<Unit, boost::heap::compare<CmpUnit> >;
	//using heap_t = boost::heap::binomal_heap<Unit, boost::heap::compare<CmpUnit> >;
	using handle_t = typename heap_t::handle_type;
	heap_t heap;
	std::unordered_map<id_t, handle_t, hash_t> khm; // key-handler mapper
};

template <class id_t, class hash_t>
id_t PrioritizedHolder<id_t, hash_t>::top(){
	return heap.top().k;
}

template <class id_t, class hash_t>
void PrioritizedHolder<id_t, hash_t>::pop(){
	id_t k = top();
	heap.pop();
	khm.erase(k);
}

template <class id_t, class hash_t>
void PrioritizedHolder<id_t, hash_t>::update(const id_t& k, const float& p){
	auto it = khm.find(k);
	if(it == khm.end()){ // new key -> push
		khm[k] = heap.push(Unit{ k, p });
	} else{ // exist key -> update
		heap.update(it->second, Unit{ k, p });
	}
}

template <class id_t, class hash_t>
void PrioritizedHolder<id_t, hash_t>::reset(const id_t& k){
	auto it = khm.find(k);
	if(it != khm.end()){
		heap.erase(it->second);
		khm.erase(it);
	}
}
