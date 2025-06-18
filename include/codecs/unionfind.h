#pragma once
#include <boost/unordered/unordered_flat_set.hpp>

struct DisjointUnionFind {

    std::vector<int32_t> parents;
    std::vector<int32_t> ranks;
    
    // map partition root to the root of all partitions that are known to be disjoint
    std::vector<boost::unordered_flat_set<int32_t>> root_to_disjoint; 

    DisjointUnionFind(size_t n) {
        parents.resize(n);
        for(int i = 0; i < n; i++) {
            parents[i] = i;
        }
        ranks.resize(n, 0);
        root_to_disjoint.resize(n);
    }

    int32_t find(int32_t key) {
        if(parents[key] != key) {
            parents[key] = find(parents[key]);
        }
        return parents[key];
    }

    bool is_disjoint(int32_t key1, int32_t key2) {
        int32_t root1 = find(key1);
        int32_t root2 = find(key2);
        if(root_to_disjoint[root1].contains(root2)) return true;
        return false;
    }

    void make_disjoint(int32_t key1, int32_t key2) {
        int32_t root1 = find(key1);
        int32_t root2 = find(key2);
        root_to_disjoint[root1].insert(root2);
        root_to_disjoint[root2].insert(root1);
    }

    bool is_union(int32_t key1, int32_t key2) {
        return find(key1) == find(key2);
    }


    void make_union(int32_t key1, int32_t key2) {
        int32_t root1 = find(key1);
        int32_t root2 = find(key2);

        if(ranks[root1] >= ranks[root2]) { // move root2 into root1
            for(const auto& other : root_to_disjoint[root2]) {
                root_to_disjoint[other].erase(root2);
                root_to_disjoint[other].insert(root1);
            }
            root_to_disjoint[root1].insert(root_to_disjoint[root2].begin(), root_to_disjoint[root2].end());
            root_to_disjoint[root2].clear();
            parents[root2] = root1;
            ranks[root1] += (ranks[root1] == ranks[root2]); // increment rank only if they had equal rank
        } else { // move root1 into root2
            for(const auto& other : root_to_disjoint[root1]) {
                root_to_disjoint[other].erase(root1);
                root_to_disjoint[other].insert(root2);
            }
            root_to_disjoint[root2].insert(root_to_disjoint[root1].begin(), root_to_disjoint[root1].end());
            root_to_disjoint[root1].clear();
            parents[root1] = root2;
        }
    }

};