#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <functional>
#include <cmath>

#include "huffman.h"
#include "bitstream.h"
#include "multicut.h"

struct EncodingResult {
    int bits_used; // how many bits the encoding consumed. can be negative because this struct allows 
    // arithmetic operations and might be used to encode differences
    float encoding_error; // the error this encoding encurs. the metric must be defined by the Codec itself

    EncodingResult& NOINLINE operator+=(const EncodingResult& other) {
        bits_used += other.bits_used;
        encoding_error += other.encoding_error;
        return *this;
    }

    friend EncodingResult NOINLINE operator+(EncodingResult left, const EncodingResult& right) {
        left += right;
        return left;
    }

    EncodingResult& NOINLINE operator-=(const EncodingResult& other) {
        bits_used -= other.bits_used;
        encoding_error -= other.encoding_error;
        return *this;
    }

    friend EncodingResult NOINLINE operator-(EncodingResult left, const EncodingResult& right) {
        left -= right;
        return left;
    }

    float NOINLINE cost(float weight_bits, float weight_err) const {
        return weight_bits * bits_used + weight_err * encoding_error;
    }

    friend std::ostream& operator<<(std::ostream& os, const EncodingResult& e) {
        os << "EncodingResult(bit_used=" << e.bits_used << ", encoding_error=" << e.encoding_error << ")";
        return os;  
    }
};

struct PartitionCodecBase {
    virtual void initialize(const std::vector<PartitionData>* partitions, const cv::Mat* img) = 0;

    virtual void write_encoding(BitStream& bs) = 0;

    virtual EncodingResult test_encoding(partition_key pk) = 0;
    virtual EncodingResult test_join_encoding(partition_key pk1, partition_key pk2) = 0;
    
    virtual void decode(BitStreamReader& bs, cv::Mat& out_img) = 0;

    virtual void notify_init(partition_key pk) = 0;
    virtual void notify_join(partition_key pk1, partition_key pk2) = 0;

    virtual std::unique_ptr<PartitionCodecBase> clone() const = 0;

    virtual ~PartitionCodecBase() = default;

};
struct MulticutCodecBase {
    virtual void write_encoding(BitStream& bs, const cv::Mat& mask) = 0;
    virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols) = 0;
    virtual std::unique_ptr<MulticutCodecBase> clone() const = 0;
    virtual ~MulticutCodecBase() = default;
};