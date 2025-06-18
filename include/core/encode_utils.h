#pragma once
#include "util.h"
#include "optimizer.h"
#include "mean_codec.h"
#include "multicut_codec.h"
#include "compressed_image.h"
#include "multicut_aware_codec.h"


class Codec {

    std::unique_ptr<AbstractOptimizer> optimizer;
    std::unique_ptr<PartitionCodecBase> partition_codec;
    std::unique_ptr<MulticutCodecBase> multicut_codec;
    bool compressed = true;

    friend class CodecBuilder;

public:

    BitStream encode_from_mask(const cv::Mat& img, const cv::Mat& mask) const {
        Multicut mc(mask);
        BitStream res;

        if(compressed) {
            CompressedMulticutImage mc_img(mask, img);
            mc_img.encode(mc, res, partition_codec->clone().get(), multicut_codec->clone().get());
        } else {
            MulticutImage mc_img(mask, img);
            mc_img.encode(mc, res, partition_codec->clone().get(), multicut_codec->clone().get());
        }
        return res;
    }

    BitStream encode_from_multicut(const cv::Mat& img, const Multicut& mc) const {
        return encode_from_mask(img, mc.mask);
    }

    BitStream optimize_encode(const cv::Mat& img) const {
        auto mc = optimize(img);
        return encode_from_mask(img, mc.mask);
    };

    Multicut optimize(const cv::Mat& img) const {
        return optimize(img, MulticutImage::get_default_mask(img, 1));
    } 

    Multicut optimize(const cv::Mat& img, const cv::Mat& mask) const {
        return optimizer->optimize(img, mask);
    }

    BitStream optimize_encode(const cv::Mat& img, const cv::Mat& mask) const {
        auto mc = optimize(img, mask);
        return encode_from_mask(img, mc.mask);
    };

    std::pair<cv::Mat, size_t> optimize_and_get_mask_with_size(const cv::Mat& img) const {
        Multicut mc = optimize(img);
        BitStream bs;
        multicut_codec->write_encoding(bs, mc.mask);
        return std::make_pair(mc.mask, bs.size());
    }

    std::unique_ptr<MulticutImage> decode(const BitStream& bs) const {
        if(compressed) {
            return std::make_unique<CompressedMulticutImage>(bs, partition_codec.get(), multicut_codec.get());
        }
        else {
            return std::make_unique<MulticutImage>(bs, partition_codec.get(), multicut_codec.get());
        }
    }
    
};

struct CodecBuilder {

private:
    Codec codec;

public:

    CodecBuilder(const CodecBuilder&) = delete;
    CodecBuilder& operator=(const CodecBuilder&) = delete;
    CodecBuilder(CodecBuilder&&) = default;
    CodecBuilder() = default;

    template<typename PartitionCodec, typename... Args>
    CodecBuilder& set_partition_codec(Args&&... args) {
        codec.partition_codec = std::make_unique<PartitionCodec>(std::forward<Args>(args)...);
        return *this;
    }

    template<typename MulticutCodec, typename... Args>
    CodecBuilder& set_multicut_codec(Args&&... args) {
        codec.multicut_codec = std::make_unique<MulticutCodec>(std::forward<Args>(args)...);
        return *this;
    }

    // the following bools are used to simplify the interface, to the builder,
    // by making it possible to set an optimizer without a PartitionCodec.
    // This is somewhat redundant, as the partition codec needs to be set anyway for encoding and
    // decoding purposes.
    template <typename Optimizer, typename... Args>
    static constexpr bool TakesCodecParam =
        std::is_same_v<Optimizer, GreedyOptimizer> || std::is_same_v<Optimizer, GreedyGridOptimizer>;

    template <typename Optimizer, typename... Args>
    static constexpr bool IsValidOptimizer = std::is_constructible_v<Optimizer, Args...>;

    // This templates accepts all instantiations that provide correct parameters for the Optimizer.
    template<typename Optimizer, typename... Args,
             std::enable_if_t<IsValidOptimizer<Optimizer, Args...>, int> = 0>
    CodecBuilder& set_optimizer(Args&&... args) {
        codec.optimizer = std::make_unique<Optimizer>(std::forward<Args>(args)...);
        return *this;
    }

    // If a substition to the above template fails, attempt to make the definition whole by 
    // reusing a previously declared partition_codec as a parameter.
    // This is only valid if set_optimizer has previously been called and will throw a runtime error otherwise.
    template<typename Optimizer, typename... Args,
             std::enable_if_t<TakesCodecParam<Optimizer, Args...> && !IsValidOptimizer<Optimizer, Args...>, int> = 0>
    CodecBuilder& set_optimizer(Args&&... args) {
        if(!codec.partition_codec) {
            throw std::runtime_error(
                "Unable to use set_optimizer: No partition_codec found. "
                "Set one using set_partition_codec or explicitly specify one in set_optimizer."
            );
        }
        codec.optimizer = std::make_unique<Optimizer>(std::forward<Args>(args)..., std::move(codec.partition_codec->clone()));
        return *this;
    }

    CodecBuilder& enable_compression() {
        codec.compressed = true;
        return *this;
    }

    CodecBuilder& disable_compression() {
        codec.compressed = false;
        return *this;
    }

    Codec create() {
        assert(codec.partition_codec);
        assert(codec.multicut_codec);
        return std::move(codec);
    }



};


