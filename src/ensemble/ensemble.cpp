#include <iostream>
#include "ensemble.h"
#include "train.h"
#include "encode_utils.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/split_data.hpp>

namespace ensemble {

    std::vector<std::unique_ptr<MulticutCodecBase>> make_configs() {
        std::vector<std::unique_ptr<MulticutCodecBase>> res;
        res.push_back(std::make_unique<DefaultMulticutCodec>());
        res.push_back(std::make_unique<DynamicHuffmanCodec>());
        res.push_back(std::make_unique<BorderCodec>());
        res.push_back(std::make_unique<MulticutAwareCodec>(std::make_unique<BlockCodecFactory>(4, 12), std::make_unique<BlockCodecFactory>(4, 12)));
        res.push_back(std::make_unique<MulticutAwareCodec>(std::make_unique<BlockCodecFactory>(4, 12), std::make_unique<NaiveCodecFactory>()));
        res.push_back(std::make_unique<MulticutAwareCodec>(std::make_unique<BlockCodecFactory>(8, 16), std::make_unique<NaiveCodecFactory>()));
        res.push_back(std::make_unique<MulticutAwareCodec>(std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)));
        res.push_back(std::make_unique<MulticutAwareCodec>(std::make_unique<AdapativeBitwiseCodecFactory>(2048, 2), std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)));
        return res;
    };

    
    std::vector<std::unique_ptr<MulticutCodecBase>> Configs::configs = make_configs();
    
    std::map<std::string, double> make_features(
        const cv::Mat& mask, 
        double optimization_level
    ) {
        
        std::map<std::string, double> features;
        features.emplace("rows", mask.rows);
        features.emplace("cols", mask.cols);
        features.emplace("pixels", mask.rows * mask.cols);
        features.emplace("optimization_level", optimization_level);
    
        std::unordered_map<int32_t, uint32_t> counter;
        for(uint32_t r = 0; r < mask.rows; r++) {
            for(uint32_t c = 0; c < mask.cols; c++) {
                int32_t p = mask.at<int32_t>(r, c);
                counter[p]++;
            }
        }
        
        std::vector<uint32_t> counts;
        counts.reserve(counter.size());
        double total = 0;
        for(const auto& [k, v] : counter) {
            counts.push_back(v);
            total += v;
        }
        std::sort(counts.begin(), counts.end());
    
        const uint32_t NBUCKETS = 32;
        std::vector<int32_t> buckets(NBUCKETS, 0); // support partition sizes up to 2^32 - 1 (deliberately way too large)
        
        double mean = total / counts.size();
        double var = 0.0;
    
        for(const auto& v : counts) {
            var += (v - mean) * (v - mean);
            buckets[uint32_t(std::floor(std::log2(v)))]++;
        }
        
        features.emplace("num_partition", counts.size());
        features.emplace("mean_partition_size", counts[counts.size() / 2]);
        features.emplace("avg_partition_size", mean);
        features.emplace("var_partition_size", var);
        features.emplace("std_partition_size", std::sqrt(var));
    
        for(int i = 0; i < NBUCKETS; i++) {
            features.emplace("logbucket_" + std::to_string(i), buckets[i]);
        }
    
        return features;
    };
    
    
    std::string make_key(const std::unique_ptr<MulticutCodecBase>& codec) {
    
        std::string res;
    
        if(dynamic_cast<const DefaultMulticutCodec*>(codec.get())) {
            res = "DefaultMulticutCodec";
        }
        else if(dynamic_cast<const DynamicHuffmanCodec*>(codec.get())) {
            res = "DynamicHuffmanCodec";
        }
        else if(const auto* mc_codec = dynamic_cast<const MulticutAwareCodec*>(codec.get())) {
            res = "MulticutAwareCodec[row=";
            if(dynamic_cast<const NaiveCodecFactory*>(mc_codec->row_codec_factory.get())) {
                res += "naive";
            }
            else if(const auto* c = dynamic_cast<const BlockCodecFactory*>(mc_codec->row_codec_factory.get())) {
                res += std::format("block({}|{})", c->block_size, c->freq_precision);
            }
            else if(const auto* c = dynamic_cast<const AdapativeBitwiseCodecFactory*>(mc_codec->row_codec_factory.get())) {
                res += std::format("adaptive({}|{})", c->order, c->window_size);
            }
            res += ";col=";
            if(dynamic_cast<const NaiveCodecFactory*>(mc_codec->col_codec_factory.get())) {
                res += "naive";
            }
            else if(const auto* c = dynamic_cast<const BlockCodecFactory*>(mc_codec->col_codec_factory.get())) {
                res += std::format("block({}|{})", c->block_size, c->freq_precision);
            }
            else if(const auto* c = dynamic_cast<const AdapativeBitwiseCodecFactory*>(mc_codec->col_codec_factory.get())) {
                res += std::format("adaptive({}|{})", c->order, c->window_size);
            }
            res += "]";
        }
        else if(dynamic_cast<const BorderCodec*>(codec.get())) {
            res = "BorderCodec";
        }
        else throw;
    
        return res;
    
    };
    
    std::unordered_map<std::string, int> init_key_to_index() {
        std::unordered_map<std::string, int> res;
        for(int i = 0; i < Configs::configs.size(); i++) {
            res.emplace(make_key(Configs::configs[i]), i);
        }
        return res;
    }
    
    std::unordered_map<std::string, int> Configs::key_to_index = init_key_to_index();
    
    void preprocess_data(
        const std::string& data_dir,
        const std::string& out_dir,
        const std::string& prefix,
        double optimization_level,
        uint32_t cell_size,
        bool differential_codec
    ) {
    
        auto img_paths = util::find_imgs(data_dir);
    
        std::srand(42);
        int id = std::rand();
        std::ofstream outfile(out_dir + std::format("/{}-data-{}-{}.csv", prefix, int(optimization_level), id));
    
    
        outfile << "avg_partition_size,pixels,optimization_level,img_id,img_path";
        for(int j = 0; j < Configs::configs.size(); j++) {
            outfile << "," << make_key(Configs::configs[j]);
        }
        outfile << std::endl;
    
        GreedyGridOptimizer opt(1.0, optimization_level, cell_size, std::make_unique<MeanCodec>());
    
        int proc = 0;
    
        #pragma omp parallel for schedule(dynamic, 1)
        for(int i = 0; i < img_paths.size(); i++) {
            cv::Mat img = cv::imread(img_paths[i], cv::IMREAD_COLOR);
            
            #pragma omp critical(stdio)
            {
                proc++;
                std::cout << std::format("opt {} starting to process {} ({}/{})", optimization_level, img_paths[i], proc, img_paths.size()) << std::endl;
            }

            Multicut mc = opt.optimize(img, MulticutImage::get_default_mask(img, 1));
    
            auto features = make_features(mc.mask, optimization_level);
    
            std::vector<uint32_t> bits;
            for(int j = 0; j < Configs::configs.size(); j++) {
                BitStream tmp;
                Configs::configs[j]->write_encoding(tmp, mc.mask);
                bits.push_back(tmp.size());
            }
    
            #pragma omp critical(fileio)
            {
                outfile << std::format(
                    "{},{},{},{},{}", 
                    features["avg_partition_size"], features["pixels"], features["optimization_level"], i, img_paths[i]
                );
                for(const auto& v : bits) {
                    outfile << "," << v;
                }
                outfile << std::endl;
            }
    
        }
    
        outfile.close();
    
    }
    
    void eval_model(const DecisionTree& tree) {

        auto [train_data, train_bitcosts, train_labels] = load_data(
            "../ensemble_data/train_fixed", default_target_labels
        );
        auto [test_data, test_bitcosts, test_labels] = load_data(
            "../ensemble_data/test_fixed", default_target_labels
        );
    
        arma::Row<size_t> train_preds;
        tree.Classify(train_data, train_preds);
        
        mlpack::Accuracy acc;
        std::cout << "train acc: " << acc.Evaluate(tree, train_data, train_labels) << std::endl;
        std::cout << "test acc: " << acc.Evaluate(tree, test_data, test_labels) << std::endl;
    
        arma::Row<size_t> test_preds;
        tree.Classify(test_data, test_preds);
    
        for(int l = 0; l < default_target_labels.size(); l++) {
            std::cout << std::format(
                "Naive encoding with {} costs {}", 
                default_target_labels[l],
                naive_cost(test_bitcosts, l)
            ) << std::endl;
        }
        std::cout << std::format("Encoding with model prediction costs {}", model_cost(test_bitcosts, test_preds)) << std::endl;
        
    }
    
    DecisionTree load_or_train_model(bool eval) {
        std::filesystem::path model_path("../ensemble/tree_model.json");
        
        DecisionTree model;
        if(std::filesystem::exists(model_path)) {
            std::cout << "Attempting to load model from disk..." << std::endl;
            mlpack::data::Load(model_path.string(), "tree_model", model, false);
        }
        else {
            std::cout << "Retraining model..." << std::endl;
            auto [train_data, train_bitcosts, train_labels] = load_data(
                "../ensemble_data/train_fixed", default_target_labels
            );
            auto [test_data, test_bitcosts, test_labels] = load_data(
                "../ensemble_data/test_fixed", default_target_labels
            );
        
            model = train_tree(train_data, train_labels, default_target_labels);
            mlpack::data::Save("../ensemble/tree_model.json", "tree_model", model, false);
        }

        if(eval) eval_model(model);
        return model;
    }

    std::vector<std::string> default_target_labels = {
        "BorderCodec",
        "MulticutAwareCodec[row=adaptive(4|4096);col=adaptive(2|512)]",
    };
    DecisionTree default_model = load_or_train_model(true);


    
}



