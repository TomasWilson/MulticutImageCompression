#pragma once
#include "multicut_image.h"
#include "bitstream.h"
#include "timing.h"
#include "codec.h"
#include "diagnostics.h"

#include <zlib.h>

using namespace util;

struct CompressedMulticutImage : MulticutImage {

    CompressedMulticutImage() = default;

    CompressedMulticutImage(cv::Mat mask, cv::Mat img) : MulticutImage(mask, img) {};

    CompressedMulticutImage(const CompressedMulticutImage& other) : MulticutImage(mask.clone(), img) {};

    CompressedMulticutImage(
        const BitStream& stream,
        PartitionCodecBase* partition_codec,
        MulticutCodecBase* multicut_codec) {
            BitStreamReader reader(stream);
            uLongf uncompressed_length = reader.read32u();
            BitStream compressed_stream = reader.read_substream(reader.bs.size() - reader.head);
            std::vector<unsigned char> compressed_data = compressed_stream.as_uchar(0);
    
            unsigned char* uncompressed_buf = new unsigned char[uncompressed_length];
            int status = uncompress(uncompressed_buf, &uncompressed_length, compressed_data.data(), compressed_data.size());
    
            std::vector<unsigned char> uncompressed_data(uncompressed_buf, uncompressed_buf + uncompressed_length);
            BitStream uncompressed_stream;
            uncompressed_stream.append_uchar(uncompressed_data);
            delete uncompressed_buf;
    
            MulticutImage result = MulticutImage(uncompressed_stream, partition_codec, multicut_codec);
            this->img = result.img;
            this->mask = result.mask;
        };

    virtual void encode(
        Multicut &multicut,
        BitStream& out_stream,
        PartitionCodecBase* partition_codec,
        MulticutCodecBase* multicut_codec
    )
    {
        BitStream uncompressed;
        MulticutImage::encode(multicut, uncompressed, partition_codec, multicut_codec);
        uncompressed.pad_to_bytes();

        pprintln("uncompressed size: ", uncompressed.size());

        std::vector<unsigned char> uncompressed_data = uncompressed.as_uchar(0);
        unsigned long buf_size = uncompressed_data.size() * 2 + 20;
        unsigned char* buf = new unsigned char[buf_size];

        int status = compress(buf, &buf_size, uncompressed_data.data(), uncompressed_data.size());
        std::vector<unsigned char> compressed_data(buf, buf + buf_size);
        out_stream.append<uint32_t>(uncompressed_data.size(), 32);
        out_stream.append_uchar(compressed_data);
        delete buf;

        DIAGNOSTICS_MESSAGE("compressed_multicut_image_bits", out_stream.size());
    }

    virtual std::unique_ptr<MulticutImage> clone() const {
        return std::make_unique<CompressedMulticutImage>(*this);
    }


};