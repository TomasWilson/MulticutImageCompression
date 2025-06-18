#include "multicut_codec.h"

cv::Mat mask_from_edges(std::vector<bool>& row_edges, std::vector<bool>& col_edges, size_t rows, size_t cols) {

    const int32_t MAX = std::numeric_limits<int32_t>::max();

    cv::Mat mask = cv::Mat::ones(rows, cols, CV_32SC1) * MAX;

    auto row_edge_exists = [&](size_t row, size_t col) -> bool {
        if(row < 0 || row >= rows) return false;
        if(col < 0 || col >= cols - 1) return false;
        return row_edges[row * (cols-1) + col];
    };

    auto col_edge_exists = [&](size_t row, size_t col) -> bool {
        if(row < 0 || row >= rows - 1) return false;
        if(col < 0 || col >= cols) return false;
        return col_edges[col * (rows-1) + row];
    };

    int32_t index = -1; // the segment that is currently being added

    for(size_t r = 0; r < rows; r++) {
        for(size_t c = 0; c < cols; c++) {

            if(mask.at<int32_t>(r, c) != MAX) continue;
            
            std::vector<int32_t> row_stack;
            row_stack.push_back(r);

            std::vector<int32_t> col_stack;
            col_stack.push_back(c);

            index++;
            mask.at<int32_t>(r, c) = index;

            while(!row_stack.empty()) {

                size_t cr = row_stack.back();
                row_stack.pop_back();

                size_t cc = col_stack.back();
                col_stack.pop_back();

                if(row_edge_exists(cr, cc) && (mask.at<int32_t>(cr, cc+1) == MAX)) {
                    row_stack.push_back(cr);
                    col_stack.push_back(cc+1);
                    mask.at<int32_t>(cr, cc+1) = index;
                }
                if(row_edge_exists(cr, cc-1) && (mask.at<int32_t>(cr, cc-1) == MAX)) {
                    row_stack.push_back(cr);
                    col_stack.push_back(cc-1);
                    mask.at<int32_t>(cr, cc-1) = index;
                }
                if(col_edge_exists(cr, cc) && (mask.at<int32_t>(cr+1, cc) == MAX)) {
                    row_stack.push_back(cr+1);
                    col_stack.push_back(cc);
                    mask.at<int32_t>(cr+1, cc) = index;
                }
                if(col_edge_exists(cr-1, cc) && (mask.at<int32_t>(cr-1, cc) == MAX)) {
                    row_stack.push_back(cr-1);
                    col_stack.push_back(cc);
                    mask.at<int32_t>(cr-1, cc) = index;
                }
            }
        }
    }
    return mask;

}


void DefaultMulticutCodec::write_encoding(BitStream& bs, const cv::Mat& mask) {
    // append row edges
    for(size_t r = 0; r < mask.rows; r++) {
        for(size_t c = 0; c < mask.cols - 1; c++) {
            bs.append(mask.at<int32_t>(r, c) == mask.at<int32_t>(r, c+1), 1);
        }
    }

    // append col edges
    for(size_t c = 0; c < mask.cols; c++) {
        for(size_t r = 0; r < mask.rows - 1; r++) {
            bs.append(mask.at<int32_t>(r, c) == mask.at<int32_t>(r+1, c), 1);
        }
    }
}


cv::Mat DefaultMulticutCodec::read_mask(BitStreamReader& reader, size_t rows, size_t cols) {
    size_t n_row_edges = (cols - 1) * rows;
    size_t n_col_edges = cols * (rows - 1);

    auto row_edges = reader.read_bits(n_row_edges);
    auto col_edges = reader.read_bits(n_col_edges);

    return mask_from_edges(row_edges, col_edges, rows, cols);
}

std::unique_ptr<MulticutCodecBase> DefaultMulticutCodec::clone() const {
    return std::make_unique<DefaultMulticutCodec>(*this);
}