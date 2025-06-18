#include "util.h"

namespace util {
    
    std::pair<cv::Mat, size_t> slico_segment(cv::Mat img, int region_size, float compactness) {
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = 
            cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLICO, region_size, compactness);

        slic->iterate(20);
        slic->enforceLabelConnectivity();

        cv::Mat labels;
        slic->getLabels(labels);
        size_t n_labels = slic->getNumberOfSuperpixels();

        return std::make_pair(labels, n_labels);
    }

    cv::Mat relabel(const cv::Mat& mask) {

        cv::Mat res = mask.clone();
        std::unordered_map<int32_t, int32_t> m;

        for(int r = 0; r < mask.rows; r++) {
            for(int c = 0; c < mask.cols; c++) {
                int32_t old_key = res.at<int32_t>(r, c);
                if(m.find(old_key) == m.end()) {
                    m[old_key] = m.size();
                }
                res.at<int32_t>(r, c) = m.at(old_key);
            }
        }

        return res;

    }

    cv::Mat display_mask(const cv::Mat& mask) {

        srand(42); // ensure displayed mask is always the same

        cv::Mat display(mask.rows, mask.cols, CV_8UC3);

        std::unordered_map<int, cv::Vec3b> key2color;

        for(size_t r = 0; r < mask.rows; r++) {
            for(size_t c = 0; c < mask.cols; c++) {
                int pk = mask.at<int>(r, c);
                cv::Vec3b p_color;
                if(key2color.find(pk) == key2color.end()) {
                    p_color = cv::Vec3b(rand()%256, rand()%256, rand()%256);
                    key2color.insert({pk, p_color});
                }
                else p_color = key2color.at(pk);
                display.at<cv::Vec3b>(r, c) = p_color;
            }
        }

        return display;
    }

    std::vector<std::string> find_imgs(const std::string& dir) {

        namespace fs = std::filesystem;
    
        std::vector<std::string> images;
        
        try {
            if (!fs::exists(dir) || !fs::is_directory(dir)) {
                throw std::runtime_error("Invalid directory: " + dir);
            }
            
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".png") {
                    images.push_back(fs::absolute(entry.path()).string());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        
        return images;
    }

    // from https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
    std::vector<std::string> readCSVRow(const std::string &row) {
        CSVState state = CSVState::UnquotedField;
        std::vector<std::string> fields {""};
        size_t i = 0; // index of the current field
        for (char c : row) {
            switch (state) {
                case CSVState::UnquotedField:
                    switch (c) {
                        case ',': // end of field
                                  fields.push_back(""); i++;
                                  break;
                        case '"': state = CSVState::QuotedField;
                                  break;
                        default:  fields[i].push_back(c);
                                  break; }
                    break;
                case CSVState::QuotedField:
                    switch (c) {
                        case '"': state = CSVState::QuotedQuote;
                                  break;
                        default:  fields[i].push_back(c);
                                  break; }
                    break;
                case CSVState::QuotedQuote:
                    switch (c) {
                        case ',': // , after closing quote
                                  fields.push_back(""); i++;
                                  state = CSVState::UnquotedField;
                                  break;
                        case '"': // "" -> "
                                  fields[i].push_back('"');
                                  state = CSVState::QuotedField;
                                  break;
                        default:  // end of quote
                                  state = CSVState::UnquotedField;
                                  break; }
                    break;
            }
        }
        return fields;
    }


}

