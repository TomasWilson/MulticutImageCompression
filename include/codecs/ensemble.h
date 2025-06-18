#pragma once
#include <string>
#include <unordered_map>
#include <iostream>

#include "codec.h"
#include "multicut_codec.h"
#include "multicut_aware_codec.h" 
#include "compressed_image.h"

#include "timing.h"
#include "optimizer.h"
#include "mean_codec.h"
#include "diagnostics.h"
#include "train.h"

#include <omp.h>

namespace ensemble {

    struct Configs {

        // see ensemble.cpp for initialization
        static std::vector<std::unique_ptr<MulticutCodecBase>> configs;
        
        static std::unordered_map<std::string, int> key_to_index;
    
        static std::unique_ptr<MulticutCodecBase>& get_codec(const std::string& key) {
            return configs[key_to_index.at(key)];
        }
    
    };

    std::map<std::string, double> make_features(
        const cv::Mat& mask, 
        double optimization_level
    );
    
    std::string make_key(const std::unique_ptr<MulticutCodecBase>& codec);
    
    
    void preprocess_data(
        const std::string& data_dir,
        const std::string& out_dir,
        const std::string& prefix,
        double optimization_level,
        uint32_t cell_size,
        bool differential_codec
    );
    
    
    extern std::vector<std::string> default_target_labels;
    extern DecisionTree default_model;

    class EnsembleCodec : public MulticutCodecBase {
        
        float optimization_level;
        std::unique_ptr<MulticutAwareCodec> mca;
        std::unique_ptr<BorderCodec> bc;

    public:
        EnsembleCodec(float optimization_level) : 
            optimization_level(optimization_level),
            mca(std::make_unique<MulticutAwareCodec>(std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), std::make_unique<AdapativeBitwiseCodecFactory>(512, 2))),
            bc(std::move(std::make_unique<BorderCodec>())) {


        }

        virtual void write_encoding(BitStream& bs, const cv::Mat& mask) {
            auto features = make_features(mask, optimization_level);
            std::vector<double> data = {features["avg_partition_size"], features["pixels"], optimization_level};
            int pred = default_model.Classify(data);
            bs.append<uint8_t>(pred, 1);        

            if(pred == 0) {
                bc->write_encoding(bs, mask);
            }
            else {
                mca->write_encoding(bs, mask);
            }
        }

        virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols) {
            bool pred = reader.read_bit();
            if(pred == 0) return bc->read_mask(reader, rows, cols);
            else return mca->read_mask(reader, rows, cols);
        }

        virtual std::unique_ptr<MulticutCodecBase> clone() const {
            return std::make_unique<EnsembleCodec>(optimization_level);
        }


    };

    // TODO FIXME
    // BitStream tree_compress(
    //     CompressedMulticutImage mc_img,
    //     float optimization_level, 
    //     std::unique_ptr<PartitionCodec> partition_codec
    // ) {
    //     // predict best codec label
    //     auto features = make_features(mc_img.mask, optimization_level);
    //     std::vector<double> data = {features["avg_partition_size"], features["pixels"], optimization_level};
    //     int pred = default_model.Classify(data);
        
    //     // get codec object
    //     auto str_label = default_target_labels[pred];
    //     auto codec = DynamicMulticutCodec(std::move(Configs::get_codec(str_label)));
    //     auto codec_idx = Configs::key_to_index.at(str_label);

    //     // encode image using codec
    //     BitStream stream;
    //     stream.append<uint8_t>(codec_idx, 8);
    //     mc_img.encode(mc_img.mask, stream, partition_codec, codec);

    //     return stream;
    // }

    // BitStream tree_compress(
    //     CompressedMulticutImage mc_img,
    //     float optimization_level
    // ) {
    //     PartitionCodec codec;
    //     return tree_compress(mc_img, optimization_level, codec);
    // }

    // MulticutImage tree_decompress(
    //     const BitStream& bs,
    //     std::unique_ptr<PartitionCodec> partition_codec
    // ) {
    //     BitStreamReader reader(bs);
    //     uint8_t codec_idx = reader.read8u();
    //     auto codec = Configs::configs.at(codec_idx);
    //     return CompressedMulticutImage::decode(reader, partition_codec, *codec);
    // }

    // MulticutImage tree_decompress(
    //     const BitStream& bs
    // ) {
    //     PartitionCodec codec;
    //     return tree_decompress(bs, codec);
    // }
    
}