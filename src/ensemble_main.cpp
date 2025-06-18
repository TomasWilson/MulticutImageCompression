#include "ensemble.h"


int main() {

    using namespace ensemble;

    auto [train_data, train_bitcosts, train_labels] = load_data(
        "../ensemble_data/train_fixed", default_target_labels
    );

    auto [test_data, test_bitcosts, test_labels] = load_data(
        "../ensemble_data/test_fixed", default_target_labels
    );


    preprocess_data("../data/splitimages/test", "../ensemble_data/test", "test-v1", 1, 128, true);
    for(float lvl = 5; lvl <= 100; lvl += 5) {
        preprocess_data("../data/splitimages/test", "../ensemble_data/test", "test-v1", lvl, 128, true);
    }

    preprocess_data("../data/splitimages/train", "../ensemble_data/train", "train-v1", 1, 128, true);
    for(float lvl = 5; lvl <= 100; lvl += 5) {
        preprocess_data("../data/splitimages/train", "../ensemble_data/train", "train-v1", lvl, 128, true);
    }

}