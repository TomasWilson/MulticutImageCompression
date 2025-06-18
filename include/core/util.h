#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/slic.hpp>

#include <unordered_map>
#include <cstdlib>
#include <cmath>
#include <filesystem>

#include "multicut.h"


namespace util {
    
    inline void pprintln() {
        #ifndef NO_DEBUG_PRINTS
        std::cout << std::endl;
        #endif
    }

    template <typename Arg, typename... Args>
    void pprint(Arg arg, Args... args)
    {
        #ifndef NO_DEBUG_PRINTS
        std::cout << arg;
        ((std::cout << " " << args), ...);
        #endif
    }

    template <typename Arg, typename... Args>
    void pprintln(Arg arg, Args... args)
    {
        #ifndef NO_DEBUG_PRINTS
        std::cout << arg;
        ((std::cout << " " << args), ...);
        std::cout << std::endl;
        #endif
    }

    std::pair<cv::Mat, size_t> slico_segment(cv::Mat img, int region_size, float compactness);

    cv::Mat relabel(const cv::Mat& mask);

    cv::Mat display_mask(const cv::Mat& mask);

    template<typename T>
    double entropy(std::vector<T> _freqs) {
        std::vector<double> freqs(_freqs.begin(), _freqs.end());

        double s = 0.0f;
        for(double f : freqs)
            s += f;
        
        double entropy = 0.0f;
        for(double f : freqs) {
            double p = f / s;
            entropy += p * std::log2(p);
        }

        return -entropy;
    }

    template<typename MAP>
    double map_entropy(const MAP& map) {
        std::vector<typename MAP::mapped_type> data;
        for(const auto& [k, v] : map) {
            data.push_back(v);
        }
        return entropy(data);
    }

    std::vector<std::string> find_imgs(const std::string& dir);

    // from https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
    enum class CSVState {
        UnquotedField,
        QuotedField,
        QuotedQuote
    };
    std::vector<std::string> readCSVRow(const std::string &row);
    ////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    size_t argmin(std::vector<T> data, T maxv) {
        size_t mini = 0;
        T minv = maxv;
        for(size_t i = 0; i < data.size(); i++) {
            if(data[i] < minv) {
                mini = i;
                minv = data[i];
            }
        }
        return mini;
    }

    inline size_t argmin(std::vector<int> data) {
        return argmin(data, std::numeric_limits<int>::max());
    }

}

