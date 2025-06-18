#pragma once
#include "multicut_image.h"
#include "compressed_image.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <iostream>


#include "unordered_set"
#include "timing.h"
#include "optimizer.h"
#include "util.h"
#include "mean_codec.h"
#include "multicut_codec.h"
#include "multicut_aware_codec.h"
#include "encode_utils.h"
#include "ensemble.h"

void fill(cv::Mat& in, cv::Mat& out, int r, int c, int fill_target, int fill_index) {
    if(out.at<int32_t>(r, c) != -1 || in.at<int32_t>(r, c) != fill_target) return;
    out.at<int32_t>(r, c) = fill_index;
    if(r > 0) fill(in, out, r-1, c, fill_target, fill_index);
    if(r < in.rows - 1) fill(in, out, r+1, c, fill_target, fill_index);
    if(c > 0) fill(in, out, r, c-1, fill_target, fill_index);
    if(c < in.cols - 1) fill(in, out, r, c+1, fill_target, fill_index);
}


// the algo seems to not return connected regions (necessarily)
// this simple flood fill solution fixes that.
cv::Mat fix_mask(cv::Mat mask) {

    cv::Mat res(mask.rows, mask.cols, CV_32SC1, cv::Scalar(-1));
    int idx = 1;

    for(size_t r = 0; r < mask.rows; r++) {
        for(size_t c = 0; c < mask.cols; c++) {
            if (res.at<int32_t>(r, c) == -1) {
                fill(mask, res, r, c, mask.at<int32_t>(r, c), idx);
                idx++;
            } 
        }
    }

    return res;
}


// void test_with_grid() {

//     // cv::Mat img = cv::imread("../data/images/icon_512/actions-address-book-new.png");
//     cv::Mat img = cv::imread("../data/images/photo_wikipedia/001.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/031.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/013.png");
    
//     // cv::Mat img = cv::imread("../data/images/pngimg/towel_PNG99.png");
//     // cv::Mat img = cv::imread("../data/images/icon_64/actions-address-book-new.png");
//     // cv::Mat img = cv::imread("../data/images/pngimg/chef_PNG37.png");
//     // cv::Mat img = cv::imread("../data/images/pngimg/academy_awards_PNG9.png");
//     // cv::Mat img = cv::imread("../data/images/screenshot_game/3dbrick2.png");
//     // cv::Mat img = cv::imread("../data/kodim05_128.png");
//     // cv::Mat img = cv::imread("../data/images/photo_kodak/kodim01.png");
//     // cv::Mat img = cv::imread("../data/images/screenshot_web/en.wikipedia.org.png");

//     // cv::Mat img = cv::imread("../data/images/icon_64/status-network-idle.png");

//     cv::Mat default_mask = MulticutImage::get_default_mask(img, 1);

//     tic();

//     CompressedMulticutImage multicut_image(
//         default_mask, 
//         img
//     );

//     // optimize(multicut_image, 1, 5);
//     // optimize2(multicut_image, 1, 10);
//     // Multicut mc = optimize_greedy_joining<MeanCodec>(multicut_image, 1, 5, true);
    
//     auto mean_codec = MeanCodec();
//     std::function<Multicut(const MulticutImage&, float, float, bool, MeanCodec)> fn = optimize_greedy_joining<MeanCodec>;
//     Multicut mc = optimize_grid(fn, multicut_image, 1, 20, 128, mean_codec);

//     mean_codec = MeanCodec();
//     auto mca = MulticutAwareCodec(
//         std::make_unique<AdapativeBitwiseCodecFactory>(4096, 8),
//         std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
//     );
//     BitStream encoded_stream;
//     multicut_image.encode(mc, encoded_stream, mean_codec, mca);

//     toc("Time to encode:");

//     pprintln("enc stream size:", encoded_stream.size());
//     size_t encoded_size_in_kib = encoded_stream.size() / 8 / 1024;
//     pprintln("encoded image size:", encoded_size_in_kib, "kiB");

//     auto uncompressed_img = CompressedMulticutImage::decode(encoded_stream, MeanCodec(), MulticutAwareCodec(
//         std::make_unique<AdapativeBitwiseCodecFactory>(4096, 8),
//         std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
//         // std::make_unique<BlockCodecFactory>(6, 12),
//         // std::make_unique<BlockCodecFactory>(4, 12)
//     ));

//     toc("Time to encode-decode:");
//     std::cout << DIAGNOSTICS_GET("optimizer_duration_ms", int) << std::endl;

//     size_t mc_bits = *DIAGNOSTICS_GET("multicut_bits", size_t);
//     size_t mc_uncompressed = *DIAGNOSTICS_GET("multicut_image_encoded_bits", size_t);

//     double frac = double(mc_bits) / double(mc_uncompressed);
//     std::cout << "ratio of multicut to entire size (uncompressed): " << frac << std::endl;

//     // cv::imwrite("address-book.png", img);
//     // cv::imwrite("address-book-decoded.png", uncompressed_img.img);
//     // cv::imwrite("address-book-mask.png", display_mask(mc.mask));

//     #ifndef PROFILING
//     cv::Mat img_decoded = uncompressed_img.img;
    
//     cv::imshow("Original image", img);
//     cv::imshow("Decoded image", img_decoded);
//     cv::imshow("Decoded Mask", display_mask(uncompressed_img.mask));
//     #endif

//     cv::waitKey(0);

// }

void test_border() {

    // cv::Mat img = cv::imread("../data/images/icon_512/actions-address-book-new.png");
    cv::Mat img = cv::imread("../data/images/photo_wikipedia/001.png");
    // cv::Mat img = cv::imread("../data/images/photo_wikipedia/031.png");
    // cv::Mat img = cv::imread("../data/images/photo_wikipedia/013.png");
    
    // cv::Mat img = cv::imread("../data/images/pngimg/towel_PNG99.png");
    // cv::Mat img = cv::imread("../data/images/icon_64/actions-address-book-new.png");
    // cv::Mat img = cv::imread("../data/images/pngimg/chef_PNG37.png");
    // cv::Mat img = cv::imread("../data/images/pngimg/academy_awards_PNG9.png");
    // cv::Mat img = cv::imread("../data/images/screenshot_game/3dbrick2.png");
    // cv::Mat img = cv::imread("../data/images/photo_kodak/kodim01.png");
    // cv::Mat img = cv::imread("../data/images/screenshot_web/en.wikipedia.org.png");

    // cv::Mat img = cv::imread("../data/images/icon_64/status-network-idle.png");


    tic();

    float compression_lvl = 10.0f;

    auto codec = CodecBuilder().set_partition_codec<MeanCodec>()
                               .set_optimizer<GreedyGridOptimizer>(1, compression_lvl, 128)
                               .create();

                            
    auto mask = codec.optimize(img).mask;

    std::cout << "computed mask" << std::endl;

    auto c1 = CodecBuilder().set_multicut_codec<DynamicHuffmanCodec>()
                            .set_partition_codec<MeanCodec>()
                            .create();

    auto c2 = CodecBuilder().set_multicut_codec<BorderCodec>(true)
                            .set_partition_codec<MeanCodec>()
                            .create();

    auto dec1 = c1.encode_from_mask(img, mask);
    auto mcimg1 = c1.decode(dec1);

    size_t mc_bits = *DIAGNOSTICS_GET("multicut_bits", size_t);
    size_t mc_uncompressed = *DIAGNOSTICS_GET("multicut_image_encoded_bits", size_t);
    std::cout << "enc bits: " << mc_bits << " unenc bits: " << mc_uncompressed << std::endl;
    double frac = double(mc_bits) / double(mc_uncompressed);
    std::cout << "ratio of multicut to entire size (uncompressed): " << frac << std::endl;


    auto dec2 = c2.encode_from_mask(img, mask);
    auto mcimg2 = c2.decode(dec2);

    mc_bits = *DIAGNOSTICS_GET("multicut_bits", size_t);
    mc_uncompressed = *DIAGNOSTICS_GET("multicut_image_encoded_bits", size_t);
    std::cout << "enc bits: " << mc_bits << " unenc bits: " << mc_uncompressed << std::endl;
    frac = double(mc_bits) / double(mc_uncompressed);
    std::cout << "ratio of multicut to entire size (uncompressed): " << frac << std::endl;

    cv::Mat diff;
    cv::absdiff(mcimg1->mask, mcimg2->mask, diff);
    bool areEqual = (cv::countNonZero(diff) == 0);
    std::cout << "mats are equal? " << areEqual << std::endl;

    cv::imshow("Original image", img);
    cv::imshow("dec1 mask", display_mask(mcimg1->mask));
    cv::imshow("dec2 mask", display_mask(mcimg2->mask));
    cv::imshow("dec1 img", mcimg1->img);
    cv::imshow("dec2 img", mcimg2->img);

    // #ifndef PROFILING
    // cv::Mat img_decoded = uncompressed_img->img;
    
    // cv::imshow("Decoded image", img_decoded);
    // cv::imshow("Decoded Mask", display_mask(uncompressed_img->mask));
    // #endif

    cv::waitKey(0);

}


// void test_lossless() {

//     // cv::Mat img = cv::imread("../data/images/icon_512/actions-address-book-new.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/001.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/031.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/013.png");
    
//     // cv::Mat img = cv::imread("../data/images/pngimg/towel_PNG99.png");
//     // cv::Mat img = cv::imread("../data/images/icon_64/actions-address-book-new.png");
//     // cv::Mat img = cv::imread("../data/images/pngimg/chef_PNG37.png");
//     // cv::Mat img = cv::imread("../data/images/pngimg/academy_awards_PNG9.png");
//     // cv::Mat img = cv::imread("../data/images/screenshot_game/3dbrick2.png");
//     cv::Mat img = cv::imread("../data/images/photo_kodak/kodim01.png");
//     // cv::Mat img = cv::imread("../data/images/screenshot_web/en.wikipedia.org.png");

//     // cv::Mat img = cv::imread("../data/images/icon_64/status-network-idle.png");

//     cv::Mat default_mask = MulticutImage::get_default_mask(img, 1);

//     tic();

//     CompressedMulticutImage multicut_image(
//         default_mask, 
//         img
//     );

//     // Multicut mc2 = optimize_lossless(multicut_image);
//     // Multicut mc = Multicut(fix_mask(mc2.mask));

//     Multicut mc = optimize_lossless(multicut_image);

//     BitStream encoded_stream;
//     multicut_image.encode(mc, encoded_stream, DifferentialMeanCodec(), BorderCodec());

//     toc("Time to encode:");

//     pprintln("enc stream size:", encoded_stream.size());
//     size_t encoded_size_in_kib = encoded_stream.size() / 8 / 1024;
//     pprintln("encoded image size:", encoded_size_in_kib, "kiB");

//     auto uncompressed_img = CompressedMulticutImage::decode(encoded_stream, DifferentialMeanCodec(), BorderCodec());

//     toc("Time to encode-decode:");
//     std::cout << DIAGNOSTICS_GET("optimizer_duration_ms", int) << std::endl;

//     size_t mc_bits = *DIAGNOSTICS_GET("multicut_bits", size_t);
//     size_t mc_uncompressed = *DIAGNOSTICS_GET("multicut_image_encoded_bits", size_t);

//     double frac = double(mc_bits) / double(mc_uncompressed);
//     std::cout << "ratio of multicut to entire size (uncompressed): " << frac << std::endl;

//     cv::Mat img_decoded = uncompressed_img.img;
    
//     cv::imshow("Original image", img);
//     cv::imshow("Decoded image", img_decoded);
//     cv::imshow("Decoded Mask", display_mask(uncompressed_img.mask));

//     cv::waitKey(0);

// }

// void test_differential() {

//     // cv::Mat img = cv::imread("../data/images/icon_512/actions-address-book-new.png");
//     cv::Mat img = cv::imread("../data/images/photo_wikipedia/001.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/031.png");
//     // cv::Mat img = cv::imread("../data/images/photo_wikipedia/013.png");

//     cv::Mat default_mask = MulticutImage::get_default_mask(img, 1);

//     tic();

//     CompressedMulticutImage multicut_image(
//         default_mask, 
//         img
//     );

//     std::function<Multicut(const MulticutImage&, float, float, bool, DifferentialMeanCodec)> fn = optimize_greedy_joining<DifferentialMeanCodec>;
//     Multicut mc = optimize_grid(fn, multicut_image, 1, 20, 128, DifferentialMeanCodec());

//     BitStream encoded_stream;
//     multicut_image.encode(mc, encoded_stream, DifferentialMeanCodec(), MulticutAwareCodec(
//         // std::make_unique<AdapativeBitwiseCodecFactory>(4096, 8),
//         // std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
//         std::make_unique<BlockCodecFactory>(6, 12),
//         std::make_unique<BlockCodecFactory>(4, 12)
//     ));

//     toc("Time to encode:");

//     pprintln("enc stream size:", encoded_stream.size());
//     size_t encoded_size_in_kib = encoded_stream.size() / 8 / 1024;
//     pprintln("encoded image size:", encoded_size_in_kib, "kiB");

//     auto uncompressed_img = CompressedMulticutImage::decode(encoded_stream, DifferentialMeanCodec(), MulticutAwareCodec(
//         // std::make_unique<AdapativeBitwiseCodecFactory>(4096, 8),
//         // std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
//         std::make_unique<BlockCodecFactory>(6, 12),
//         std::make_unique<BlockCodecFactory>(4, 12)
//     ));

//     toc("Time to encode-decode:");
//     std::cout << DIAGNOSTICS_GET("optimizer_duration_ms", int) << std::endl;

//     size_t mc_bits = *DIAGNOSTICS_GET("multicut_bits", size_t);
//     size_t mc_uncompressed = *DIAGNOSTICS_GET("multicut_image_encoded_bits", size_t);

//     double frac = double(mc_bits) / double(mc_uncompressed);
//     std::cout << "ratio of multicut to entire size (uncompressed): " << frac << std::endl;

//     #ifndef PROFILING
//     cv::Mat img_decoded = uncompressed_img.img;
    
//     cv::imshow("Original image", img);
//     cv::imshow("Decoded image", img_decoded);
//     cv::imshow("Decoded Mask", display_mask(uncompressed_img.mask));
//     #endif

//     cv::waitKey(0);

// }