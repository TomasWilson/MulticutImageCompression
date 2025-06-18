#include "codec.h"
#include "optimizer.h"
#include "mean_codec.h"
#include "multicut_codec.h"
#include "compressed_image.h"
#include "multicut_aware_codec.h"
#include "encode_utils.h"
#include "ensemble.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/extract.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace bindings {

    np::ndarray mat_to_ndarray(const cv::Mat& mat) {
        if (mat.channels() == 1) {
            np::dtype dtype = np::dtype::get_builtin<int32_t>();
            np::ndarray ndarray = np::empty(bp::make_tuple(mat.rows, mat.cols), dtype);
            std::memcpy(ndarray.get_data(), mat.data, mat.total() * sizeof(int32_t));
            return ndarray;
        } else {
            np::dtype dtype = np::dtype::get_builtin<unsigned char>();
            np::ndarray ndarray = np::empty(bp::make_tuple(mat.rows, mat.cols, mat.channels()), dtype);
            std::memcpy(ndarray.get_data(), mat.data, mat.total() * mat.elemSize());
            return ndarray;
        }
    }
    
    cv::Mat ndarray_to_mat(const np::ndarray& arr) {
        if (arr.get_nd() == 2) {
            return cv::Mat(arr.shape(0), arr.shape(1), CV_32SC1, arr.get_data());
        } else if (arr.get_nd() == 3) {
            int channels = arr.shape(2);
            return cv::Mat(arr.shape(0), arr.shape(1), CV_8UC(channels), arr.get_data());
        } else {
            throw std::runtime_error("Unsupported ndarray shape. Expected 2D or 3D.");
        }
    }

    enum MULTICUT_CODEC {
        HUFFMAN,
        BORDER,
        MULTICUT_AWARE,
        ENSEMBLE
    };

    enum PARTITION_CODEC {
        SIMPLE,
        DIFFERENTIAL
    };

    enum OPTIMIZER {
        LOSSLESS,
        GREEDY,
        GREEDY_GRID
    };


    bp::tuple make_mask(
        const np::ndarray& img, 
        MULTICUT_CODEC m_codec,
        PARTITION_CODEC p_codec,
        OPTIMIZER optim,
        float optim_level
    ) {

        auto cb = CodecBuilder();

        switch(p_codec) {
            case SIMPLE: cb.set_partition_codec<MeanCodec>(); break;
            case DIFFERENTIAL: cb.set_partition_codec<DifferentialMeanCodec>(); break;
        };

        switch(m_codec) {
            case HUFFMAN: cb.set_multicut_codec<DynamicHuffmanCodec>(); break;
            case BORDER: cb.set_multicut_codec<BorderCodec>(); break;
            case MULTICUT_AWARE: cb.set_multicut_codec<MulticutAwareCodec>(
                    std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), 
                    std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
                ); break;
            case ENSEMBLE: cb.set_multicut_codec<ensemble::EnsembleCodec>(optim_level); break;
        }

        switch(optim) {
            case LOSSLESS: cb.set_optimizer<LosslesOptimizer>(); break;
            case GREEDY: cb.set_optimizer<GreedyOptimizer>(1.0, optim_level, true); break;
            case GREEDY_GRID: cb.set_optimizer<GreedyGridOptimizer>(1.0, optim_level, 128); break;
        }

        Codec c = cb.create();
        auto p = c.optimize_and_get_mask_with_size(ndarray_to_mat(img));

        return bp::make_tuple(mat_to_ndarray(p.first), p.second); // return mask, mask_bits
    }

    bp::tuple encode_mask(
        const np::ndarray& img, 
        const np::ndarray& mask, 
        MULTICUT_CODEC m_codec,
        PARTITION_CODEC p_codec,
        bool entropy_compress,
        float optim_level
    ) {

        auto cb = CodecBuilder();

        switch(p_codec) {
            case SIMPLE: cb.set_partition_codec<MeanCodec>(); break;
            case DIFFERENTIAL: cb.set_partition_codec<DifferentialMeanCodec>(); break;
        };

        switch(m_codec) {
            case HUFFMAN: cb.set_multicut_codec<DynamicHuffmanCodec>(); break;
            case BORDER: cb.set_multicut_codec<BorderCodec>(); break;
            case MULTICUT_AWARE: cb.set_multicut_codec<MulticutAwareCodec>(
                    std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), 
                    std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
                ); break;
            case ENSEMBLE: cb.set_multicut_codec<ensemble::EnsembleCodec>(optim_level); break;
        }

        if(!entropy_compress) cb.disable_compression();

        Codec c = cb.create();

        auto _img = ndarray_to_mat(img);
        auto _mask = ndarray_to_mat(mask);
        BitStream enc = c.encode_from_mask(_img, _mask);
        auto dec = c.decode(enc);

        return bp::make_tuple(mat_to_ndarray(dec->img), enc.size()); // return decoded img, encoded size

    }

    bp::tuple encode_decode_mask(
        const np::ndarray& img, 
        MULTICUT_CODEC m_codec,
        PARTITION_CODEC p_codec
    ) {
        return bp::make_tuple(0, 0); // return decoded img, img_bits
    }


    // compress and decompress an image given a codec,
    // return the decompressed image and the size used to store it in compressed form
    bp::tuple compress_decompress(
        const np::ndarray& img, 
        Codec& codec
    ) {
        cv::Mat _img = ndarray_to_mat(img);
        auto bs = codec.optimize_encode(_img);
        auto dec_img = codec.decode(bs);
        return bp::make_tuple(mat_to_ndarray(dec_img->img), bs.size());
    }

    np::ndarray optimize_grid_mean(
        const np::ndarray& img, 
        float compression_strength, 
        unsigned cell_size
    ) {
        auto opt = GreedyGridOptimizer(1.0, compression_strength, cell_size, std::make_unique<MeanCodec>());
        auto _img = ndarray_to_mat(img);
        auto mc = opt.optimize(_img, MulticutImage::get_default_mask(_img, 1));
        return mat_to_ndarray(mc.mask);
    }

    size_t test_huffman_encoding(const np::ndarray& mask) {
        BitStream bs;
        auto enc = DynamicHuffmanCodec();
        enc.write_encoding(bs, ndarray_to_mat(mask));
        return bs.size();
    }

    size_t test_adaptive_multicut_aware_encoding(
        const np::ndarray& mask,
        int row_context_size,
        int row_order,
        int col_context_size,
        int col_order
    ) {
        BitStream bs;
        auto enc = MulticutAwareCodec(
                    std::make_unique<AdapativeBitwiseCodecFactory>(row_context_size, row_order), 
                    std::make_unique<AdapativeBitwiseCodecFactory>(col_context_size, col_order));
        enc.write_encoding(bs, ndarray_to_mat(mask));
        return bs.size();
    }

    size_t test_border_encoding(const np::ndarray& mask) {
        BitStream bs;
        auto enc = BorderCodec();
        enc.write_encoding(bs, ndarray_to_mat(mask));
        return bs.size();
    }

    size_t test_ensemble_encoding(const np::ndarray& mask, float optimization_level) {
        BitStream bs;
        auto enc = ensemble::EnsembleCodec(optimization_level);
        enc.write_encoding(bs, ndarray_to_mat(mask));
        return bs.size();
    }

    bp::tuple huffman_mean_grid(
        const np::ndarray& img, 
        float compression_strength, 
        unsigned cell_size
    ) {
        Codec codec = CodecBuilder()
                        .set_multicut_codec<DynamicHuffmanCodec>()
                        .set_partition_codec<MeanCodec>()
                        .set_optimizer<GreedyGridOptimizer>(1.0f, compression_strength, cell_size)
                        .create();

        return compress_decompress(img, codec);
    }

    bp::tuple adaptive_multicut_mean_grid(
        const np::ndarray& img, 
        float compression_strength, 
        unsigned cell_size
    ) {
        Codec codec = CodecBuilder()
                        .set_multicut_codec<MulticutAwareCodec>(
                            std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), 
                            std::make_unique<AdapativeBitwiseCodecFactory>(512, 2))
                        .set_partition_codec<MeanCodec>()
                        .set_optimizer<GreedyGridOptimizer>(1.0f, compression_strength, cell_size)
                        .create();

        return compress_decompress(img, codec);
    }

    bp::tuple border_mean_grid(
        const np::ndarray& img, 
        float compression_strength, 
        unsigned cell_size
    ) {
        Codec codec = CodecBuilder()
                        .set_multicut_codec<BorderCodec>()
                        .set_partition_codec<MeanCodec>()
                        .set_optimizer<GreedyGridOptimizer>(1.0f, compression_strength, cell_size)
                        .create();

        return compress_decompress(img, codec);
    }

    // Export the function to Python
    BOOST_PYTHON_MODULE(mlcv_py) {
        np::initialize();

        bp::enum_<MULTICUT_CODEC>("MULTICUT_CODEC")
        .value("HUFFMAN", HUFFMAN)
        .value("BORDER", BORDER)
        .value("MULTICUT_AWARE", MULTICUT_AWARE)
        .value("ENSEMBLE", ENSEMBLE);

        bp::enum_<PARTITION_CODEC>("PARTITION_CODEC")
            .value("SIMPLE", SIMPLE)
            .value("DIFFERENTIAL", DIFFERENTIAL);

        bp::enum_<OPTIMIZER>("OPTIMIZER")
            .value("LOSSLESS", LOSSLESS)
            .value("GREEDY", GREEDY)
            .value("GREEDY_GRID", GREEDY_GRID);


        bp::def("make_mask_with_size", make_mask, 
            (
            bp::arg("img"),
            bp::arg("multicut_codec"),
            bp::arg("partition_codec"),
            bp::arg("optimizer"),
            bp::arg("compression_strength")
            )
        );

        bp::def("encode_mask_with_size", encode_mask, (
            bp::arg("img"),
            bp::arg("mask"),
            bp::arg("multicut_codec"),
            bp::arg("partition_codec"),
            bp::arg("entropy_compress")
        ));

        /*-----------------------------------------------------------------------------------*/


        bp::def("optimize_grid_mean", optimize_grid_mean, 
            (
            bp::arg("img"),
            bp::arg("compression_strength")=1.0f,
            bp::arg("cell_size")=128
            )
        );



        bp::def("optimize_grid_mean", optimize_grid_mean, 
            (
            bp::arg("img"),
            bp::arg("compression_strength")=1.0f,
            bp::arg("cell_size")=128
            )
        );

        /*-----------------------------------------------------------------------------------*/

        bp::def("test_huffman_encoding", test_huffman_encoding, (bp::arg("mask"))); 

        bp::def("test_adaptive_multicut_aware_encoding", test_adaptive_multicut_aware_encoding, 
            (
            bp::arg("mask"),
            bp::arg("row_context_size") = 4096,
            bp::arg("row_order") = 4,
            bp::arg("col_context_size") = 512,
            bp::arg("col_order") = 2
            )
        );

        bp::def("test_border_encoding", test_border_encoding, (bp::arg("mask"))); 

        bp::def("test_ensemble_encoding", test_ensemble_encoding, (bp::arg("mask"), bp::arg("optimization_level"))); 
        
        /*-----------------------------------------------------------------------------------*/

        bp::def("huffman_mean_grid", huffman_mean_grid, 
            (
            bp::arg("img"),
            bp::arg("compression_strength")=1.0f,
            bp::arg("cell_size")=128
            )
        );
    
        bp::def("adaptive_multicut_mean_grid", adaptive_multicut_mean_grid, 
            (
            bp::arg("img"),
            bp::arg("compression_strength")=1.0f,
            bp::arg("cell_size")=128
            )
        );

    }

} 

