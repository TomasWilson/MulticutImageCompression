#include "train.h"

namespace ensemble {

    std::tuple<arma::mat, arma::mat, arma::Row<size_t>> load_data(
        const std::string& dir, 
        const std::vector<std::string>& target_labels
    ) {

        std::vector<float> partsize;
        std::vector<int> pixels;
        std::vector<int> optlevel;

        std::vector<std::vector<int>> bit_costs;
        std::vector<std::string> codecs;

        std::vector<int> label_indices;
        bool first = true;


        for(const auto& entry : std::filesystem::directory_iterator(dir)) {

            if (entry.is_regular_file() && entry.path().extension() == ".csv") {

                auto s = entry.path().string();
                std::ifstream csv(s);

                std::string line;
                std::getline(csv, line);

                // read the codec names from the first file
                if(first) {
                    auto tokens = util::readCSVRow(line);
                    for(const auto& l : target_labels) {
                        ptrdiff_t pos = std::find(tokens.begin(), tokens.end(), l) - tokens.begin();
                        if(pos < tokens.size()) {
                            label_indices.push_back(pos);
                        }
                    }
                    assert(label_indices.size() == target_labels.size());
                    first = false;
                }

                while(std::getline(csv, line)) {

                    auto tokens = util::readCSVRow(line);
                    partsize.push_back(std::stof(tokens[0]));
                    pixels.push_back(std::stoi(tokens[1]));
                    optlevel.push_back(std::stoi(tokens[2]));

                    std::vector<int> bits;
                    for(const auto& i : label_indices) {
                        bits.push_back(std::stoi(tokens[i]));
                    }
                    bit_costs.push_back(bits);
                    
                }

            }
        }

        arma::mat dataset(3, partsize.size());
        arma::mat m_bit_costs(label_indices.size(), bit_costs.size());
        arma::Row<size_t> labels(bit_costs.size());

        for(int i = 0; i < partsize.size(); i++) {
            dataset.at(0, i) = double(partsize[i]); 
            dataset.at(1, i) = double(pixels[i]); 
            dataset.at(2, i) = double(optlevel[i]); 
        }

        for(int i = 0; i < bit_costs.size(); i++) {
            labels.at(i) = size_t(util::argmin(bit_costs[i]));
            for(int j = 0; j < label_indices.size(); j++) {
                m_bit_costs.at(j, i) = bit_costs[i][j];
            }
        }

        return std::make_tuple(dataset, m_bit_costs, labels);

    }

    DecisionTree train_tree(
        const arma::mat& data, 
        const arma::Row<size_t>& labels,
        const std::vector<std::string>& target_labels
    ) {
        DecisionTree res;
        res.Train(data, labels, target_labels.size());
        return res;
    }

}