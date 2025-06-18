#pragma once
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

// #include "ensemble.h"
#include "util.h"

#include <mlpack.hpp>
#include <string>
#include <filesystem>

namespace ensemble {

    std::tuple<arma::mat, arma::mat, arma::Row<size_t>> load_data(
        const std::string& dir,
        const std::vector<std::string>& target_labels
    );
    
    using DecisionTree = mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit>;
    
    DecisionTree train_tree(
        const arma::mat& data, 
        const arma::Row<size_t>& labels,
        const std::vector<std::string>& target_labels
    );
    
    // how much it costs to always predict the algorithm indicated by the given label
    inline size_t naive_cost(
        const arma::mat& bit_costs,
        int label
    ) {
        size_t total = 0;
        for(int i = 0; i < bit_costs.n_cols; i++) {
            total += bit_costs.at(label, i);
        }
        return total;
    }
    
    inline size_t model_cost(
        const arma::mat& bit_costs,
        const arma::Row<size_t>& preds
    ) {
        size_t total = 0;
        for(int i = 0; i < bit_costs.n_cols; i++) {
            total += bit_costs.at(preds.at(i), i);
        }
        return total;
    }
}

