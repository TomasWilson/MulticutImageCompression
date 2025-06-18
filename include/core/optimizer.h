#pragma once
#include "timing.h"
#include "multicut_image.h"
#include "diagnostics.h"

#include <queue>
#include <deque>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <ctime>
#include <random>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/heap/priority_queue.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/heap/binomial_heap.hpp>

struct AbstractOptimizer {
    virtual Multicut optimize(const cv::Mat& img, const cv::Mat& mask) = 0;
    virtual ~AbstractOptimizer() = default;
};

struct LosslesOptimizer : AbstractOptimizer {

    virtual Multicut optimize(const cv::Mat& img, const cv::Mat& mask) {
        cv::Mat out_mask;
        cv::Mat tmp_mask;
        mask.copyTo(out_mask);
        mask.copyTo(tmp_mask);
    
        static std::vector<cv::Point2i> deltas = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
        partition_key k = 0;
    
        for(int r = 0; r < out_mask.rows; r++) {
            for(int c = 0; c < out_mask.cols; c++) {
    
                cv::Point2i p(c, r);
                if(tmp_mask.at<int32_t>(p) == -1) continue;
                
                std::deque<cv::Point2i> dq;
                dq.emplace_back(p);
    
                tmp_mask.at<int32_t>(p) = -1;
                out_mask.at<int32_t>(p) = k;
    
                while(!dq.empty()) {
                    
                    cv::Point2i curr = dq.front();
                    dq.pop_front();
    
                    for(const auto& d : deltas) {
                        cv::Point2i nb = curr + d;
                        if(nb.x < 0 || nb.x >= out_mask.cols) continue;
                        if(nb.y < 0 || nb.y >= out_mask.rows) continue;
                        if(tmp_mask.at<int32_t>(nb) == -1) continue;
                        if(img.at<cv::Vec3b>(curr) != img.at<cv::Vec3b>(nb)) continue;
    
                        tmp_mask.at<int32_t>(nb) = -1; // mark as (to be) visited
                        out_mask.at<int32_t>(nb) = k;
                        dq.emplace_back(nb);
                    }
    
                }
    
                k++;
    
            }
        }
    
        return Multicut(out_mask);
    }

};

struct JoinMove {
    EncodingResult gain;
    double gain_val;
    partition_key k1;
    partition_key k2;
    int t1;
    int t2;
};

struct JoinMoveComparator {
    bool operator()(const JoinMove& a, const JoinMove& b) const {
        return a.gain_val < b.gain_val;
    }
};

class RandomSet {
private:
    std::vector<partition_key> elements;
    boost::unordered_flat_map<partition_key, int> indices;
    std::mt19937 mt;
    
public:

    RandomSet(size_t n_keys) {
        elements.reserve(n_keys);
        indices.reserve(n_keys);
        for(int i = 0; i < n_keys; i++) {
            add(i);
        }
        init_mt();
    }

    RandomSet() {
        init_mt();
    }

    void init_mt() {
        int seed = 33;
        mt = std::mt19937(seed);
    }
    
    void add(partition_key x) {
        indices[x] = elements.size();
        elements.push_back(x);
    }
    
    void deleteElement(partition_key x) {
        int index = indices[x];
        partition_key lastElement = elements.back();
        
        elements[index] = lastElement;
        indices[lastElement] = index;
        
        elements.pop_back();
        indices.erase(x);
    }
    
    partition_key get() {
        return elements[mt() % elements.size()];
    }

    bool empty() {
        return elements.empty();
    }

};

// using priority_queue_impl = boost::heap::fibonacci_heap<JoinMove, boost::heap::compare<JoinMoveComparator>>;
using priority_queue_impl = boost::heap::d_ary_heap<JoinMove, boost::heap::compare<JoinMoveComparator>, boost::heap::arity<4>>;
// using priority_queue_impl = boost::heap::binomial_heap<JoinMove, boost::heap::compare<JoinMoveComparator>>;


// A "perfect" join is one, that does not increase the error and, at the same time, does not increase the amount of bits used
// in the case of mean-value coding this simply means to group identical colors together
// it's not entirely clear that such moves should always be applied (because they could potentially hurt later joins down the line)
// This function greedily searches and applies all of these joins.
// This helps processing down the line, especially if the image contains a lot of such regions.
// This is of particularily high importance if the image contains constantly colored regions and the MeanCodec is used.
inline void apply_perfect_lb_joins(
    std::vector<EncodingResult>& partition_cost,
    Multicut& mc,
    const cv::Mat& img,
    std::unique_ptr<PartitionCodecBase>& partition_codec
) {

    RandomSet rs(mc.partitions.size());

    while(!rs.empty()) {

        partition_key pk = rs.get();
        bool changed = false;
        key_set neighbours = mc.get_neighbours(pk);
        for(partition_key pk_nb : neighbours) {
            EncodingResult res = partition_codec->test_join_encoding(pk, pk_nb);
            EncodingResult gain = partition_cost[pk] + partition_cost[pk_nb] - res;

            if(gain.bits_used >= 0 && gain.encoding_error >= 0) {
                partition_key pk_join = mc.join(pk, pk_nb);
                partition_cost.at(pk_join) = partition_cost[pk] + partition_cost[pk_nb] - gain;

                if(pk_join == pk) rs.deleteElement(pk_nb);
                else rs.deleteElement(pk);

                changed = true;
                break;
            }
        }

        if(!changed) rs.deleteElement(pk);
    }

}

struct GreedyOptimizer : AbstractOptimizer {

private:
    float weight_err;
    float weight_size; 
    bool init_perfect_joins;
    std::unique_ptr<PartitionCodecBase> partition_codec;

public:

    GreedyOptimizer(
        float weight_err, 
        float weight_size, 
        bool init_perfect_joins,
        std::unique_ptr<PartitionCodecBase> partition_codec
    ) : weight_err(weight_err), 
        weight_size(weight_size), 
        init_perfect_joins(init_perfect_joins), 
        partition_codec(std::move(partition_codec)) {

    }

    virtual Multicut optimize(const cv::Mat& img, const cv::Mat& mask) {

        Multicut multicut(mask);

        auto& partitions = multicut.partitions;
        partition_codec->initialize(&partitions, &img);
    
        priority_queue_impl moves;
        moves.reserve(img.rows * img.cols * 2);
    
        // std::unordered_map<UnorderedKey, std::pair<time, time>, UnorderedKeyHash> join_age; // if or when this join was last computed
        std::vector<EncodingResult> partition_cost;
        partition_cost.resize(partitions.size());
    
        EncodingResult total_result = {0, 0.0};
    
        // compute initial costs
        for(partition_key pk = 0; pk < partitions.size(); pk++) {
            partition_codec->notify_init(pk);
            EncodingResult result = partition_codec->test_encoding(pk);
            total_result += result;
            partition_cost[pk] = result;
        }
    
        if(init_perfect_joins) {
            apply_perfect_lb_joins(partition_cost, multicut, img, partition_codec);
        }
    
        // compute initial join potential for all neighbouring partitions
        for(partition_key pk = 0; pk < partitions.size(); pk++) {
            key_set neighbours = multicut.get_neighbours(pk);
            for(partition_key pk_nb : neighbours) {
                if(pk < pk_nb) { // make sure a join is only considered once
                    
                    EncodingResult res = partition_codec->test_join_encoding(pk, pk_nb);
                    EncodingResult gain = (partition_cost[pk] + partition_cost[pk_nb]) - res;
                    float gain_val = gain.cost(weight_size, weight_err);
                    if(gain_val > 0) {
                        moves.push({gain, gain_val, pk, pk_nb, partitions.at(pk).age, partitions.at(pk_nb).age});
                    }
    
                }
            }
        }
    
        int its = 1;
        int newmoves = 0;
    
        // run greedy joining until convergence
        while(!moves.empty()) {
    
            its++;
    
            // regularily keeping the size of the pq small is beneficial for performance
            if(newmoves > 25'000) {
                newmoves = 0;
                priority_queue_impl new_pq;
                for(const auto& join : moves) {
                    if(multicut.valid_join(join.k1, join.t1, join.k2, join.t2)) {
                        new_pq.push(join);
                    }
                }
                if(new_pq.empty()) break; // forgetting this check cost me 45 minutes of debugging time
                moves = std::move(new_pq);
            }
    
            const JoinMove& best_move = moves.top();
    
            // check if move is still valid
            if (!multicut.valid_join(best_move.k1, best_move.t1, best_move.k2, best_move.t2)) {
                moves.pop();
                continue;
            }
    
            // perform the join, mark both involved partitions as "changed" and note the cost.
            partition_codec->notify_join(best_move.k1, best_move.k2);
            partition_key pk_join = multicut.join(best_move.k1, best_move.k2);
            partition_cost.at(pk_join) = partition_cost[best_move.k1] + partition_cost[best_move.k2] - best_move.gain;
            total_result -= best_move.gain;
    
            // for all neighbours of the joint partition, recompute join costs
            const auto& neighbours = multicut.get_neighbours(pk_join);
            // std::vector<partition_key> neighbours_vec;
            // neighbours_vec.reserve(neighbours.size());
            // neighbours_vec.insert(neighbours_vec.end(), neighbours.begin(), neighbours.end());
    
            std::vector<JoinMove> newMoves;
            newMoves.resize(neighbours.size());
            newmoves += neighbours.size();
    
            int i = 0;
            for(partition_key pk_nb : neighbours) {
                EncodingResult old_res = partition_cost.at(pk_join) + partition_cost.at(pk_nb);
                EncodingResult res = partition_codec->test_join_encoding(pk_join, pk_nb);
                EncodingResult gain = old_res - res;
                newMoves[i++] = {gain, gain.cost(weight_size, weight_err), pk_join, pk_nb, partitions.at(pk_join).age, partitions.at(pk_nb).age};
            }
    
            moves.pop();
    
            // std::make_heap<std::vector<JoinMove>::iterator, JoinMoveComparator>(newMoves.begin(), newMoves.end());
            for(const auto& m : newMoves) {
                if(m.gain_val > 0)
                    moves.push(m);
            }
    
    
        }
    
        return Multicut(multicut.mask); // TODO: This is broken!!!!!!

    }


};

struct GreedyGridOptimizer : AbstractOptimizer {

    float weight_size, weight_err;
    uint32_t cell_size;
    
    std::unique_ptr<PartitionCodecBase> partition_codec;

    GreedyGridOptimizer(
        float weight_err, 
        float weight_size, 
        uint32_t cell_size,
        std::unique_ptr<PartitionCodecBase> partition_codec
    ) : weight_err(weight_err), 
        weight_size(weight_size),
        cell_size(cell_size),
        partition_codec(std::move(partition_codec)) {

    }

    virtual Multicut optimize(const cv::Mat& img, const cv::Mat& mask) {

        tic(589);

        MulticutImage large_img(mask, img);
    
        int offset = 0;
    
        int cells_per_row = (img.cols - 1) / cell_size + 1; // ceil
        int cells_per_col = (img.rows - 1) / cell_size + 1;
        int n_cells = cells_per_col * cells_per_row;
    
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < n_cells; i++) {

            // #ifndef NO_DEBUG_PRINTS
            // #pragma omp critical
            // {
                //     std::cout << i << "/" << n_cells << std::endl;
                // }
            // #endif
                
            int start_c = (i % cells_per_row) * cell_size;
            int start_r = (i / cells_per_row) * cell_size;
            auto roi_rect = large_img.subarea(start_r, start_c, cell_size, cell_size);
            
            GreedyOptimizer cell_optimizer(weight_err, weight_size, true, std::move(partition_codec->clone()));
            
            MulticutImage sub_img = large_img.subimage(roi_rect);
            Multicut sub_mc = cell_optimizer.optimize(sub_img.img, sub_img.mask);
    
            cv::Mat roi = large_img.mask(roi_rect);
            sub_mc.mask.copyTo(roi);
            roi += i * cell_size * cell_size;
        }
    
        tic(123);
        GreedyOptimizer full_optimizer(weight_err, weight_size, false, std::move(partition_codec->clone()));
        auto res = full_optimizer.optimize(large_img.img, large_img.mask);
        // DIAGNOSTICS_MESSAGE("optimizer_last_call_ms", toc("last optimize_fn call took", 123));
        // DIAGNOSTICS_MESSAGE("optimizer_duration_ms", toc("optimize took", 589));
        
        return Multicut(res.mask);
    }

};