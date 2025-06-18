#pragma once

#include <unordered_map>
#include <unordered_set>

#include "multicut_codec.h"
#include "context_encoder.h"
#include "unionfind.h"

/*

Number of multicuts on a Grid:

The "MulticutAwareCodec" makes the encoding of the multicut more explicit.
In its simplest form, it performs the same operations as the "DefaultMulticutCodec",
except that it ommits reading and writing all bits that don't add any new information, 
as defined by the previously read bits.

For example, for the following Mask:

1 - 1
|   |
2 - 2

The edges 1-1 and 2-2 are clearly not cut and the other two are clearly cut. 
If these edges are read, and the left edge 1-2 is read, the other 1-2 edge 
need not be read (or wrote): By that time it is already clear to the decoder 
that 1 and 2 are not in the same partition. 

(Note that the decoder has no access to the actual labels of the partitions. 
The labels are created ascendingly in row-major order, starting at 0, as is 
required by all multicut codecs.)

*/




struct MulticutAwareCodec : public MulticutCodecBase {

    static MulticutAwareCodec row_adaptive_col_adaptive;

    std::unique_ptr<AbstractCodecFactory> row_codec_factory;
    std::unique_ptr<AbstractCodecFactory> col_codec_factory;

    MulticutAwareCodec() : 
        row_codec_factory(std::make_unique<NaiveCodecFactory>()),
        col_codec_factory(std::make_unique<NaiveCodecFactory>()) {

    }

    MulticutAwareCodec(
        std::unique_ptr<AbstractCodecFactory> row_codec_factory,
        std::unique_ptr<AbstractCodecFactory> col_codec_factory) 
        : row_codec_factory(std::move(row_codec_factory)), 
        col_codec_factory(std::move(col_codec_factory)) {

    }

    MulticutAwareCodec(const MulticutAwareCodec& other) : 
        MulticutAwareCodec(other.row_codec_factory->clone(), 
                           other.col_codec_factory->clone()) {
        
    }

    // convenience constructor, so you don't have to create the unique_ptr's as a caller. (see above)
    // instead, you just supply the desired factory types as template argument
    // only makes sense to use when the factorys dont have parameters
    template<typename RowCodecFactory, typename ColCodecFactory>
    static MulticutAwareCodec create() {
        MulticutAwareCodec codec;
        codec.row_codec_factory = std::make_unique<RowCodecFactory>();
        codec.row_codec_factory = std::make_unique<ColCodecFactory>();
        return codec;
    }

    virtual void write_encoding(BitStream& bs, const cv::Mat& mask) {

        auto row_encoder = row_codec_factory->make_encoder(bs);

        size_t n_bits = (mask.rows - 1) * mask.cols + mask.rows * (mask.cols - 1);
        DisjointUnionFind df(n_bits);

        auto make_key = [&](int r, int c) {
            return r * mask.cols + c;
        };

        std::vector<bool> ctx;
        ctx.reserve(n_bits);

        for(int r = 0; r < mask.rows; r++) {
            for(int c = 0; c < mask.cols - 1; c++) {
                if(mask.at<partition_key>(r, c) == mask.at<partition_key>(r, c+1)) {
                    df.make_union(make_key(r, c), make_key(r, c+1));
                    row_encoder->encode_bit(true, ctx);
                    ctx.push_back(true);
                }
                else {
                    df.make_disjoint(make_key(r, c), make_key(r, c+1));
                    row_encoder->encode_bit(false, ctx);
                    ctx.push_back(false);
                }
            }
        }

        row_encoder->finalize();

        auto col_encoder = col_codec_factory->make_encoder(bs);

        for(int c = 0; c < mask.cols; c++) {
            for(int r = 0; r < mask.rows - 1; r++) {
                int k1 = make_key(r, c);
                int k2 = make_key(r+1, c);
                if(df.is_disjoint(k1, k2)) {
                    ctx.push_back(false);
                    continue;
                }
                if(df.is_union(k1, k2)) {
                    ctx.push_back(true);
                    continue;
                }
                if(mask.at<partition_key>(r, c) == mask.at<partition_key>(r+1, c)) {
                    df.make_union(k1, k2);
                    col_encoder->encode_bit(true, ctx);
                    ctx.push_back(true);
                }
                else {
                    df.make_disjoint(k1, k2);
                    col_encoder->encode_bit(false, ctx);
                    ctx.push_back(false);
                }
            }
        }

        col_encoder->finalize();
    }

    virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols) {

        auto row_decoder = row_codec_factory->make_decoder(reader);
        row_decoder->initialize();

        size_t n_bits = (rows - 1) * cols + rows * (cols - 1);
        DisjointUnionFind df(n_bits);

        std::vector<bool> ctx;
        ctx.reserve(n_bits);

        size_t n_row_edges = rows * (cols - 1);
        auto make_key = [&](int r, int c) {
            return r * cols + c;
        };

        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols - 1; c++) {
                bool edge = row_decoder->decode_bit(ctx);
                if(edge) {
                    df.make_union(make_key(r, c), make_key(r, c+1));
                } 
                else {
                    df.make_disjoint(make_key(r, c), make_key(r, c+1));
                }
                ctx.push_back(edge);
            }
        }

        row_decoder->finalize();

        auto col_decoder = col_codec_factory->make_decoder(reader);
        col_decoder->initialize();

        for(int c = 0; c < cols; c++) {
            for(int r = 0; r < rows - 1; r++) {
                int k1 = make_key(r, c);
                int k2 = make_key(r+1, c);
                if(df.is_disjoint(k1, k2)) {
                    ctx.push_back(false);
                    continue;
                }
                if(df.is_union(k1, k2)) {
                    ctx.push_back(true);
                    continue;
                }

                bool edge = col_decoder->decode_bit(ctx);
                if(edge) {
                    df.make_union(k1, k2);
                }
                else {
                    df.make_disjoint(k1, k2);
                }
                ctx.push_back(edge);
            }
        }

        col_decoder->finalize();

        cv::Mat res(rows, cols, CV_32S);
        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols; c++) {
                res.at<partition_key>(r, c) = df.find(make_key(r, c));
            }
        }

        return util::relabel(res);

    }

    virtual std::unique_ptr<MulticutCodecBase> clone() const {
        return std::make_unique<MulticutAwareCodec>(*this);
    }

};

