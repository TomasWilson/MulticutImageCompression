#pragma once
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cassert>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "bitstream.h"
#include "header.h"
#include "timing.h"
#include "util.h"

#include <boost/unordered/unordered_flat_set.hpp>

// disallow inlining of certain functions while in profiling mode on GCC
// needed for more detailed profiles
#if defined(PROFILING) && defined(__GNUC__)
    #define NOINLINE __attribute__ ((noinline))
#else
    #define NOINLINE
#endif

typedef int partition_key;
using key_set = boost::unordered_flat_set<partition_key>;
// using key_set = std::unordered_set<partition_key>;

struct PartitionData
{
    std::vector<cv::Point2i> points;
    int age;
};

struct Multicut
{

    cv::Mat mask;
    std::vector<PartitionData> partitions;
    std::vector<key_set> neighbours;

    Multicut() = default;

    Multicut(const cv::Mat& mask) : mask(mask.clone())
    {
        init_from_mask<true>();
    };

    static Multicut without_relabel(const cv::Mat& mask) {
        Multicut res;
        res.mask = mask.clone();
        res.init_from_mask<false>();
        return res;
    }

    // apply a move indiciated by an edge, updating the mask and partitions.
    // returns a partition key that references the name of the new partition
    // the returned key is one of pk1 or pk2.
    partition_key join(partition_key pk1, partition_key pk2)
    {
        assert(pk1 != pk2);

        // ensure pk2 points to the larger partition
        // if (partitions.at(pk1).points.size() > partitions.at(pk2).points.size())
        if(pk2 > pk1)
        {
            std::swap(pk1, pk2);
        }

        partitions.at(pk1).age++;
        partitions.at(pk2).age++;

        auto &points1 = partitions.at(pk1).points;
        auto &points2 = partitions.at(pk2).points;

        for (const cv::Point2i &p : points1) // is this really needed?
        {
            mask.at<partition_key>(p) = pk2;
        }

        points2.reserve(points1.size() + points2.size());
        points2.insert(points2.end(), points1.begin(), points1.end());

        auto &nbs1 = neighbours.at(pk1);
        auto &nbs2 = neighbours.at(pk2);

        for (auto old_nb : nbs1)
        {
            neighbours.at(old_nb).erase(pk1);
            neighbours.at(old_nb).insert(pk2);
        }

        nbs2.insert(nbs1.begin(), nbs1.end());
        nbs2.erase(pk1);
        nbs2.erase(pk2);

        nbs1 = key_set();

        return pk2;
    }

    bool valid_join(partition_key pk1, int age1, partition_key pk2, int age2)
    {
        return partitions.at(pk1).age == age1 && partitions.at(pk2).age == age2;
    }

    const key_set& NOINLINE get_neighbours(partition_key pk) const // TODO: might wanna make this return a reference
    {
        return neighbours.at(pk);
    }

private:

    template<bool relabel>
    void init_from_mask()
    {

        if constexpr(relabel) {

            std::unordered_map<int32_t, partition_key> idx2key;
            
            for (int r = 0; r < mask.rows; r++)
            {
                for (int c = 0; c < mask.cols; c++)
                {
                    int32_t idx = mask.at<int32_t>(r, c);

                    partition_key new_key;

                    if (idx2key.find(idx) == idx2key.end())
                    {
                        new_key = partitions.size();
                        idx2key[idx] = new_key;
                        std::vector<cv::Point2i> v;
                        v.emplace_back(c, r);
                        partitions.emplace_back(v, 0);
                    }
                    else
                    {
                        new_key = idx2key.at(idx);
                        partitions[new_key].points.emplace_back(c, r);
                    }

                    mask.at<int32_t>(r, c) = new_key;
                }
            }
        }
        else {

            size_t n_partitions = mask.at<partition_key>(mask.rows-1, mask.cols-1) + 1;
            partitions.resize(n_partitions);

            for(int r = 0; r < mask.rows; r++) {
                for(int c = 0; c < mask.cols; c++) {
                    partition_key pk = mask.at<partition_key>(r, c);
                    partitions.at(pk).points.emplace_back(c, r);
                }
            }

        }

        neighbours.resize(partitions.size());

        static std::vector<std::pair<int, int>> delta = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int r = 0; r < mask.rows; r++)
        {
            for (int c = 0; c < mask.cols; c++)
            {
                partition_key pk = mask.at<int32_t>(r, c);
                for (const auto &[dr, dc] : delta)
                {
                    int nr = r + dr;
                    int nc = c + dc;
                    if (nr < 0 || nc < 0 || nr >= mask.rows || nc >= mask.cols)
                        continue;
                    partition_key nk = mask.at<int32_t>(nr, nc);
                    if (pk != nk)
                        neighbours.at(pk).insert(nk);
                }
            }
        }
    }
};
