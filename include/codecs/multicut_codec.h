#pragma once
#include <limits>

#include "codec.h"
#include "util.h"
#include "arithmetic.h"
#include "unionfind.h"

using namespace util;

cv::Mat mask_from_edges(std::vector<bool>& row_edges, std::vector<bool>& col_edges, size_t rows, size_t cols);

struct DefaultMulticutCodec : public MulticutCodecBase {
    virtual void write_encoding(BitStream& bs, const cv::Mat& mask);
    virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols);
    virtual std::unique_ptr<MulticutCodecBase> clone() const;
};

struct BlockToken {

    uint8_t data;

    bool operator==(const BlockToken& other) const {
        return data == other.data;
    }

    void encode(BitStream& bs) const {
        bs.append<uint8_t>(data, 8);
    }

    size_t encode_size() const {
        return 8;
    }

    bool operator<(const BlockToken& other) const {
        return data < other.data;
    }

    // set bit at index to v
    void set(size_t idx, bool v) {
        assert(0 <= idx && idx < 8);
        uint8_t mask = ~(uint8_t(1) << idx); // one where we dont want to change anything
        uint8_t value_mask = uint8_t(v) << idx; // one where the value is
        data = (data & mask) | value_mask; // (data & mask) clears the target bit, | value_mask sets it to v
    }

    bool get(size_t idx) const {
        assert(0 <= idx && idx < 8);
        return (data >> idx) & 0b1;
    }

    static BlockToken decode(BitStreamReader& reader) {
        BlockToken res;
        res.data = reader.read8u();
        return res;
    }

    static std::vector<BlockToken> all_blocks() {
        std::vector<BlockToken> res;
        for(int i = 0; i < 256; i++) {
            BlockToken token;
            token.data = uint8_t(i);
            res.push_back(token);
        }
        return res;
    }

};

namespace std {
    template<>
    struct hash<BlockToken> {
        std::size_t operator()(const BlockToken& token) const {
            return token.data;
        }
    };
}

// idea: encode multicut by splitting the image pixels into 2x2 blocks. Each block contains 8 outgoing edges.
// These 2^8 blocks can be encoded more efficiently using a huffman coding. For this, the frequency of each block is measured.
// From these frequencies, a huffman coder is constructed (see huffman.h). To allow for reconstruction of the code, the 256 frequencies
// are stored and transmitted together with the multicut. 
struct DynamicHuffmanCodec : MulticutCodecBase {

    const unsigned FREQ_PRECISION = 10; // TODO: constructor or template?

    virtual void write_encoding(BitStream& bs, const cv::Mat& mask) {

        std::vector<bool> row_edges;
        for(size_t r = 0; r < mask.rows; r++) {
            for(size_t c = 0; c < mask.cols - 1; c++) {
                row_edges.push_back(mask.at<int32_t>(r, c) == mask.at<int32_t>(r, c+1));
            }
        }

        std::vector<bool> col_edges;
        for(size_t c = 0; c < mask.cols; c++) {
            for(size_t r = 0; r < mask.rows - 1; r++) {
                col_edges.push_back(mask.at<int32_t>(r, c) == mask.at<int32_t>(r+1, c));
            }
        }

        auto get = [](std::vector<bool>& v, int i) -> bool {
            if(i < v.size()) return v[i];
            return false;
        };

        int edges_per_row = mask.cols - 1;
        int edges_per_col = mask.rows - 1;

        std::vector<BlockToken> tokens;
        std::map<BlockToken, size_t> token_freq;

        for(size_t r = 0; r < mask.rows; r+=2) {
            for(size_t c = 0; c < mask.cols; c+=2) {

                int row_edge_start = r * edges_per_row + c;
                int col_edge_start = c * edges_per_col + r;

                BlockToken token;
                token.set(0, get(row_edges, row_edge_start));
                token.set(1, get(row_edges, row_edge_start+1));
                token.set(2, get(row_edges, row_edge_start+edges_per_row));
                token.set(3, get(row_edges, row_edge_start+edges_per_row+1));
                token.set(4, get(col_edges, col_edge_start));
                token.set(5, get(col_edges, col_edge_start+1));
                token.set(6, get(col_edges, col_edge_start+edges_per_col));
                token.set(7, get(col_edges, col_edge_start+edges_per_col+1));

                tokens.push_back(token);
                token_freq[token]++;
            }
        }

        // normalize token freqs
        unsigned MAX_ENCODE = (1 << FREQ_PRECISION) - 1;
        unsigned max_freq = 1;

        std::vector<double> freqs;

        for(auto& [token, freq] : token_freq) {
            if(freq > max_freq) max_freq = freq;
            freqs.push_back(freq);
        }

        // pprintln("theoretical entropy:", entropy(freqs));

        for(auto& [token, freq] : token_freq) {
            // ensure that nonzero frequencies are maintained (otherwise the codec will not recognize these tokens)
            double prob = double(freq) / double(max_freq);
            freq = std::clamp(unsigned(prob * MAX_ENCODE), unsigned(1), MAX_ENCODE);
        }

        // write frequencies to stream
        for(int i = 0; i < 256; i++) {
            BlockToken key;
            key.data = uint8_t(i);
            if(token_freq.find(key) != token_freq.end()) {
                unsigned f = token_freq[key];
                bs.append<unsigned>(f, FREQ_PRECISION);
            }
            else {
                bs.append<unsigned>(0, FREQ_PRECISION);
            }
        }

        // build the huffman codec and encode all tokens to the stream
        std::vector<std::pair<BlockToken, unsigned>> v_token_freqs(token_freq.begin(), token_freq.end());
        HuffmanCodec<BlockToken> codec(v_token_freqs);
        codec.encode_tokens(tokens, bs);


    }

    virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols) {
        
        std::vector<std::pair<BlockToken, unsigned>> v_token_freqs;

        for(int i = 0; i < 256; i++) {
            BlockToken key;
            key.data = uint8_t(i);
            
            unsigned freq = reader.read<unsigned>(FREQ_PRECISION);
            if(freq > 0) {
                v_token_freqs.push_back(std::make_pair(key, freq));
            }
        }

        HuffmanCodec<BlockToken> codec(v_token_freqs);

        size_t n_row_edges = (cols - 1) * rows;
        std::vector<bool> row_edges(n_row_edges);

        size_t n_col_edges = cols * (rows - 1);
        std::vector<bool> col_edges(n_col_edges);

        int blocks_per_row = (cols+1) / 2;
        int blocks_per_col = (rows+1) / 2;
        int n_blocks = blocks_per_row * blocks_per_col;
        int edges_per_row = cols - 1;
        int edges_per_col = rows - 1;

        auto set = [](std::vector<bool>& v, int i, bool val) -> void {
            if(i < v.size()) v[i] = val;
        };

        for(int i = 0; i < n_blocks; i++) {
            BlockToken token = codec.read_next(reader);
            int block_r = i / blocks_per_row;
            int block_c = i % blocks_per_row;
            int r = block_r * 2;
            int c = block_c * 2;
            int row_edge_start = r * edges_per_row + c;
            int col_edge_start = c * edges_per_col + r;

            set(row_edges, row_edge_start,                  token.get(0));
            set(row_edges, row_edge_start+1,                token.get(1));
            set(row_edges, row_edge_start+edges_per_row,    token.get(2));
            set(row_edges, row_edge_start+edges_per_row+1,  token.get(3));
            set(col_edges, col_edge_start,                  token.get(4));
            set(col_edges, col_edge_start+1,                token.get(5));
            set(col_edges, col_edge_start+edges_per_col,    token.get(6));
            set(col_edges, col_edge_start+edges_per_col+1,  token.get(7));
        }

        return mask_from_edges(row_edges, col_edges, rows, cols);
    }

    virtual std::unique_ptr<MulticutCodecBase> clone() const {
        return std::make_unique<DynamicHuffmanCodec>(*this);
    }

};


struct BorderCodecSymbol {

    uint32_t data;
    uint32_t len;

    BorderCodecSymbol() : data(0), len(0) {};

    BorderCodecSymbol(uint32_t data, uint32_t len) : data(data), len(len) {};

    static BorderCodecSymbol from_data(const std::vector<bool>& data) {
        BorderCodecSymbol res;
        for(bool b : data) {
            res.append(b);
        }
        return res;
    }
    
    void append(bool b) {
        data = (data << 1) | b;
        len++;
    }

    bool get(int i) {
        return 0b1 & (data >> (len-i-1));
    }

    std::vector<bool> as_vec() {
        std::vector<bool> res;
        res.reserve(len);
        for(int i = 0; i < len; i++) 
            res.push_back(get(i));
        return res;
    }

    bool operator==(const BorderCodecSymbol& other) const = default;
};

namespace std {
    template<>
    struct hash<BorderCodecSymbol> {
        std::size_t operator()(const BorderCodecSymbol& s) const {
            return s.data << 16 ^ s.len;
        }
    };
}

class BorderCodecSymbolTable {

    std::unordered_map<unsigned, SimpleFrequencyTable> len2tab;
    int freq_precision;

public:

    BorderCodecSymbolTable(BitStreamReader& reader, int freq_precision=10) : freq_precision(freq_precision) {

        uint8_t ntabs = reader.read8u();
        for(int i = 0; i < ntabs; i++) {
            std::vector<uint32_t> freqs;
            uint8_t len = reader.read8u();
            for(int j = 0; j < (0b1 << len); j++) {
                uint32_t freq = reader.read<uint32_t>(freq_precision);
                freqs.push_back(freq);
            }
            len2tab.emplace(len, SimpleFrequencyTable(freqs));
        }

    }

    BorderCodecSymbolTable(const std::vector<BorderCodecSymbol>& syms, int freq_precision=10) : freq_precision(freq_precision) {
        std::unordered_map<unsigned, std::vector<unsigned>> len2data2count;
        for(const auto& sym: syms) {
            if(len2data2count.find(sym.len) == len2data2count.end()) {
                len2data2count.emplace(sym.len, std::vector<unsigned>(0b1 << sym.len, 0));
            }
            len2data2count[sym.len][sym.data]++;
        }

        unsigned max_freq = (0b1 << freq_precision) - 1;


        for(auto& [len, data2count] : len2data2count) {

            unsigned max_encode_freq = *std::max_element(data2count.begin(), data2count.end());

            for(unsigned& f : data2count) {
                if(f > 0) {
                    f = std::clamp(
                        unsigned(double(f) / double(max_encode_freq) * double(max_freq)),
                        unsigned(1),
                        max_freq
                    );
                }
            }

            len2tab.emplace(len, SimpleFrequencyTable(data2count)); 
        }
    }

    void encode(BitStream& bs) const {
        bs.append<uint8_t>(len2tab.size(), 8);
        for(const auto& [len, tab] : len2tab) {
            bs.append<uint8_t>(len, 8);
            for(int i = 0; i < (0b1 << len); i++) {
                bs.append<uint32_t>(tab.get(i), freq_precision);
            }
        }
    }

    std::vector<bool> read_symbol(WrappedArithmeticDecoder& dec, int len) {
        uint32_t data = dec.read(len2tab.at(len));
        BorderCodecSymbol sym(data, len);
        return sym.as_vec();
    }

    void write_symbol(WrappedArithmeticEncoder& enc, const BorderCodecSymbol& sym) {
        enc.write(len2tab.at(sym.len), sym.data);
    }

    void print() {
        std::cout << "--- BorderCodecSymbolTable ---" << std::endl;
        for(auto& [len, tab] : len2tab) {
            std::cout << "len = " << len << " | ";
            for(int i = 0; i < tab.getSymbolLimit(); i++) {
                std::cout << tab.get(i) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "------------------------------" << std::endl;
    }

};


struct BorderCodec : MulticutCodecBase {

    bool ENCODE_JOIN_EDGES;

    BorderCodec(bool enc_join_edges=false) : ENCODE_JOIN_EDGES(enc_join_edges) {

    }

private:
    struct Point {
        int r, c;
        
        Point(int a, int b) : r(a), c(b) {};

        bool operator==(const Point& other) const = default;
        
        int hash() const {
            return (r << 16) ^ c;
        }

        Point operator+(const Point& rhs) const {
            return Point(r + rhs.r, c + rhs.c);
        }
    };

    struct Edge {
        Point a;
        Point b;

        Edge(Point a, Point b) : a(a), b(b) {};
        
        bool operator==(const Edge& other) const {
            return (a == other.a && b == other.b) || (a == other.b && b == other.a);
        }
    };

    struct EdgeHasher {
        int operator()(const Edge& e) const {
            return e.a.hash() + e.b.hash();
        }
    };


    struct State {

        static const inline std::vector<Point> delta = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        std::unordered_map<Edge, bool, EdgeHasher> known_edges;

        int rows, cols;

        State(int rows, int cols) : rows(rows), cols(cols) {

        }

        bool valid(const Point& p) const {
            return p.r > 0 && p.c > 0 && p.r < rows && p.c < cols;
        }

        std::vector<Edge> adjacent(const Point& p) {
            std::vector<Edge> res;
            for(const auto& d : delta) {
                Point nb = p + d;
                if(!valid(nb) && !valid(p)) continue;
                Edge e(p, nb);
                if(known_edges.find(e) != known_edges.end()) continue;
                res.push_back(e);
            }
            return res;
        }

        void iterate(
            const Point& start, 
            const std::function<std::vector<bool>(std::vector<Edge>)>& read_fn
        ) {

            std::vector<Point> next;
            next.push_back(start);

            while(!next.empty()) {
                
                Point current = next.back();
                next.pop_back();

                std::vector<Edge> adj = adjacent(current);
                if(adj.size() == 0) continue;

                std::vector<bool> edge_data = read_fn(adj);

                for(int i = 0; i < adj.size(); i++) {
                    bool b = edge_data[i];
                    if(b) next.push_back(adj[i].b);
                    known_edges[adj[i]] = b;
                }
                
            }

        }

    };

    bool read_from_mask(const Edge& e, const cv::Mat& mask) {
        const Point& a = e.a;
        const Point& b = e.b;
        bool v;

        if(a.r == b.r) { // horizontal edge
            int col = std::min(a.c, b.c);
            v = (mask.at<int32_t>(a.r-1, col) == mask.at<int32_t>(a.r, col)) == ENCODE_JOIN_EDGES;
        }
        else if(a.c == b.c) { // vertical edge
            int row = std::min(a.r, b.r);
            v = (mask.at<int32_t>(row, a.c-1) == mask.at<int32_t>(row, a.c)) == ENCODE_JOIN_EDGES;
        }
        else {
            std::cout << "WARNING: Unexpected edge received." << std::endl;
        }

        return v;
    }




public:

    virtual void write_encoding(BitStream& bs, const cv::Mat& mask) {

        State s(mask.rows, mask.cols);
        std::vector<BorderCodecSymbol> syms;

        std::function<std::vector<bool>(std::vector<Edge>)> f = [&](std::vector<Edge> read_edges) {
            std::vector<bool> data;

            for(const Edge& e : read_edges) {
                bool v = read_from_mask(e, mask);
                data.push_back(v);
            }

            syms.push_back(BorderCodecSymbol::from_data(data));
            return data;
        };


        std::vector<Point> roots;
        for(int r = 0; r <= mask.rows; r++) {
            for(int c = 0; c <= mask.cols; c++) {
                Point p(r, c);
                auto adj = s.adjacent(p);
                if(std::any_of(adj.begin(), adj.end(), [&](const Edge& e) { return read_from_mask(e, mask); })) {
                    roots.push_back(p);
                    s.iterate(p, f);
                }
            }
        }

        bs.append<uint32_t>(roots.size(), 16);
        for(const auto& p : roots) {
            bs.append<uint32_t>(p.r, 16);
            bs.append<uint32_t>(p.c, 16);
        }

        BorderCodecSymbolTable tab(syms, 10);
        tab.encode(bs);

        WrappedArithmeticEncoder enc;
        for(const auto& sym : syms) {
            tab.write_symbol(enc, sym);
        }
        enc.finish(bs);
    }

    virtual cv::Mat read_mask(BitStreamReader& reader, size_t rows, size_t cols) {

        std::vector<Point> roots;
        int n = reader.read16u();
        for(int i = 0; i < n; i++) {
            int r = reader.read16u();
            int c = reader.read16u();
            Point p(r, c);
            roots.push_back(p);
        }
        
        BorderCodecSymbolTable tab(reader);

        WrappedArithmeticDecoder dec(reader);
        State s(rows, cols);

        std::function<std::vector<bool>(std::vector<Edge>)> f = [&](std::vector<Edge> read_edges) {
            return tab.read_symbol(dec, read_edges.size());
        };

        for(Point r : roots) {
            s.iterate(r, f);
        }

        auto make_key = [](int r, int c, int stride) {
            return r * stride + c;
        };

        std::vector<bool> row_edges;
        row_edges.reserve(rows * (cols-1));

        std::vector<bool> col_edges;
        col_edges.reserve((rows-1) * cols);

        // append row edges
        for(size_t r = 0; r < rows; r++) {
            for(size_t c = 0; c < cols - 1; c++) {
                Point a(r, c+1);
                Point b(r+1, c+1);
                Edge e(a, b);
                if(s.known_edges.find(e) == s.known_edges.end()) 
                    row_edges.push_back(!ENCODE_JOIN_EDGES);
                else 
                    row_edges.push_back(s.known_edges.at(e) == ENCODE_JOIN_EDGES);
            }
        }

        // append col edges
        for(size_t c = 0; c < cols; c++) {
            for(size_t r = 0; r < rows - 1; r++) {
                Point a(r+1, c);
                Point b(r+1, c+1);
                Edge e(a, b);
                if(s.known_edges.find(e) == s.known_edges.end()) 
                    col_edges.push_back(!ENCODE_JOIN_EDGES);
                else 
                    col_edges.push_back(s.known_edges.at(e) == ENCODE_JOIN_EDGES);
            }
        }

        auto res = mask_from_edges(row_edges, col_edges, rows, cols);
        return util::relabel(res);
    }

    virtual std::unique_ptr<MulticutCodecBase> clone() const {
        return std::make_unique<BorderCodec>(*this);
    }

};