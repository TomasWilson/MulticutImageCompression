#pragma once
#include "header.h"
#include "multicut.h"
#include "codec.h"
#include "util.h"
#include "diagnostics.h"

struct MulticutImage
{

    cv::Mat mask;
    cv::Mat img;

    MulticutImage() = default;
    MulticutImage(cv::Mat mask, cv::Mat img) : img(img), mask(mask.clone()) {};
    MulticutImage(const MulticutImage& other) : img(img), mask(mask.clone()) {};
    
    MulticutImage(
        const BitStream& stream,
        PartitionCodecBase* partition_codec,
        MulticutCodecBase* multicut_codec)
    {

        BitStreamReader reader(stream);
        Header header(reader);
        mask = multicut_codec->read_mask(reader, header.rows, header.cols);
        img = cv::Mat(header.rows, header.cols, CV_8UC3);
        
        Multicut mc(mask);
        partition_codec->initialize(&mc.partitions, &img);
        partition_codec->decode(reader, img);
    }

    MulticutImage(MulticutImage &&mc_img) = default;

    virtual ~MulticutImage() = default;

    virtual void encode(
        Multicut &multicut,
        BitStream& out_stream,
        PartitionCodecBase* partition_codec,
        MulticutCodecBase* multicut_codec)
    {
        partition_codec->initialize(&multicut.partitions, &img);
        const cv::Mat &mask = multicut.mask;
        Header(mask.rows, mask.cols).encode(out_stream);

        multicut_codec->write_encoding(out_stream, mask);
        DIAGNOSTICS_MESSAGE("multicut_bits", out_stream.size());
        partition_codec->write_encoding(out_stream);
        DIAGNOSTICS_MESSAGE("multicut_image_encoded_bits", out_stream.size());
    }

    int rows() const
    {
        return img.rows;
    }

    int cols() const
    {
        return img.cols;
    }

    cv::Rect subarea(int start_r, int start_c, int delta_r, int delta_c) const
    {
        auto res = cv::Rect(start_c, start_r, std::min(cols() - start_c, delta_c), std::min(rows() - start_r, delta_r));
        return res;
    }

    MulticutImage subimage(const cv::Rect &roi) const
    {
        MulticutImage result;
        result.img = img(roi);
        result.mask = mask(roi);
        return result;
    }

    // the default mask simply constructs 8x8 blocks (or smaller if not possible)
    static cv::Mat get_default_mask(const cv::Mat &img, size_t block_size = 8)
    {
        cv::Mat mask(img.rows, img.cols, CV_32SC1);
        size_t idx = 0;

        for (size_t r = 0; r < img.rows; r += block_size)
        {
            for (size_t c = 0; c < img.cols; c += block_size)
            {

                size_t dr = std::min(img.rows - r, block_size);
                size_t dc = std::min(img.cols - c, block_size);
                cv::Rect block(c, r, dc, dr);
                mask(block) = idx;
                idx++;
            }
        }

        return mask;
    }

    virtual std::unique_ptr<MulticutImage> clone() const {
        return std::make_unique<MulticutImage>(*this);
    }

};