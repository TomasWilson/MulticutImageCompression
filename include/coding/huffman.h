#pragma once
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <optional>
#include <memory>
#include <cmath>
#include <algorithm>
#include <queue>
#include <functional>
#include <utility>

#include "bitstream.h"

template <typename T>
concept EncodableTokenConcept = requires(T a, T b, BitStream& bs, BitStreamReader& reader) {
    { a == b } -> std::convertible_to<bool>;
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
    { a.encode(bs) } -> std::same_as<void>;
    { a.encode_size() } -> std::convertible_to<std::size_t>;
    { T::decode(reader) } -> std::same_as<T>;
};

template<int zero_bits, int data_bits>
struct RunlengthToken {
    
    uint32_t nzeros;
    int32_t data;

    enum Type {
        DEFAULT,
        END
    } type;

    RunlengthToken() : type(END), nzeros(0), data(0) {

    }

    RunlengthToken(uint32_t nzeros, int32_t data) : type(DEFAULT), nzeros(nzeros), data(data) {
        
    }

    bool operator==(const RunlengthToken<zero_bits, data_bits>& other) const {
        return other.type == type && other.nzeros == nzeros && other.data == data;
    }

    void encode(BitStream& bs) const {
        bs.append<uint32_t>(nzeros, zero_bits);
        bs.append<bool>(data < 0, 1);
        uint32_t payload = std::min(std::abs(data), (0b1 << (data_bits - 1)) - 1);
        bs.append<uint32_t>(payload, data_bits-1);
    }

    size_t encode_size() const {
        return zero_bits + data_bits;
    }

    static RunlengthToken<zero_bits, data_bits> decode(BitStreamReader& reader) {
        uint32_t zeros = reader.read32u(zero_bits);
        bool sign = reader.read32u(1);
        uint32_t payload = reader.read32u(data_bits-1);
        int32_t data = (1 - 2 * sign) * int32_t(payload);
        return RunlengthToken<zero_bits, data_bits>(zeros, data);
    }

    static std::vector<RunlengthToken<zero_bits, data_bits>> rle_encode(const std::vector<int>& data) {

        std::vector<RunlengthToken<zero_bits, data_bits>> result;
        int zeros = 0;

        for(int i = 0; i < data.size(); i++) {
            if (data[i] == 0) {
                zeros++;
                continue;
            }
            result.emplace_back(zeros, data[i]);
            zeros = 0;
        }

        if(zeros > 0) result.emplace_back(); // emplaces END token

        return result;
    }

};

namespace std {
    template<int zero_bits, int data_bits>
    struct hash<RunlengthToken<zero_bits, data_bits>> {
        std::size_t operator()(const RunlengthToken<zero_bits, data_bits>& token) const {
            return (uint64_t(token.nzeros) << 32) | uint64_t(token.data);
        }
    };
}

template<EncodableTokenConcept EncodableToken>
class HuffmanCodec {

    struct HuffmanToken {
        uint64_t code;
        size_t code_bits;
        void write(BitStream& bs) const {
            bs.append(code, code_bits);
        }
    };

    std::unordered_map<EncodableToken, HuffmanToken> base_tokens;
    std::optional<HuffmanToken> escape_token;

    struct Node {
        std::optional<EncodableToken> token;
        uint64_t freq;
        Node* left;
        Node* right;   

        bool is_leaf() const {
            // Note: the huffman tree is always full, so left==nullptr <=> right==nullptr
            return left == nullptr; 
        }

        // an escape token is the single leaf that is not related to an EncodableToken (if such a leaf exists)
        bool is_escape() const {
            return is_leaf() && !token.has_value();
        }
    };

    std::vector<Node*> nodes;
    Node* root;

public:

    HuffmanCodec(const HuffmanCodec&) = delete;
    HuffmanCodec(HuffmanCodec&& other) = default;
    HuffmanCodec& operator=(const HuffmanCodec&) = delete;

    HuffmanCodec(
        std::vector<std::pair<EncodableToken, unsigned>>& token_freqs,
        unsigned int escape_freq=0
    ) {
        
        auto compare = [](const Node* a, const Node* b) {
            return a->freq > b->freq;
        };
        std::priority_queue<Node*, std::vector<Node*>, decltype(compare)> pq(compare);

        for(const auto& [token, freq] : token_freqs) {
            Node* n = new Node{std::make_optional(token), freq, nullptr, nullptr};
            nodes.push_back(n);
            pq.push(n);
        }

        if(escape_freq > 0) {
            Node* n = new Node{{}, escape_freq, nullptr, nullptr};
            nodes.push_back(n);
            pq.push(n);
        }

        while (pq.size() > 1) {
            Node* a = pq.top(); pq.pop();
            Node* b = pq.top(); pq.pop();
            Node* n = new Node{{}, a->freq + b->freq, a, b};
            nodes.push_back(n);
            pq.push(n);
        }

        std::function<void(Node*, uint64_t, size_t)> dfs = [&](Node* n, uint64_t code, size_t depth) -> void {
            if(!n->is_leaf()) {
                dfs(n->left, code << 1 | 0, depth+1);
                dfs(n->right, code << 1 | 1, depth+1);
            }
            else if (n->is_escape()) 
                escape_token = std::make_optional<HuffmanToken>(code, depth);
            else {
                HuffmanToken t = {code, depth};
                base_tokens.insert(std::make_pair(*(n->token), t));
            }
        };

        root = pq.top();
        dfs(root, 0, 0);

    }

    ~HuffmanCodec() {
        for(Node* n : nodes) {
            delete n;
        }
    }

    // encode sequence of tokens to stream
    void encode_tokens(const std::vector<EncodableToken>& tokens, BitStream& bs) const {

        for(const EncodableToken& t : tokens) {
            if(base_tokens.find(t) != base_tokens.end()) {
                HuffmanToken ht = base_tokens.at(t);
                ht.write(bs);
            }
            else {
                assert(escape_token.has_value());
                (*escape_token).write(bs); // first write the escape token...
                t.encode(bs); // ...then the unencoded payload
            }
        }

    }

    // count number of bits needed to encode tokens
    size_t get_encoding_size(const std::vector<EncodableToken>& tokens) const {
        size_t res = 0;

        for(const EncodableToken& t : tokens) {
            if(base_tokens.find(t) != base_tokens.end()) {
                HuffmanToken ht = base_tokens.at(t);
                res += ht.code_bits;
            }
            else {
                assert(escape_token.has_value());
                res += (*escape_token).code_bits;
                res += t.encode_size();
            }
        }

        return res;
    }
    
    EncodableToken read_next(BitStreamReader& reader) const {
        Node* current = root;
        while(!current->is_leaf()) {
            if(reader.read_bit()) current = current->right;
            else current = current->left;
        }

        if(current->is_escape()) {
            return EncodableToken::decode(reader);
        }
        return *(current->token);
    }

};



// encode sequence of integers using MPEG like rle+huffman scheme
template<int zero_bits, int data_bits>
void encode_runlength_huffman(
    const std::vector<int>& data, 
    BitStream& bs, 
    const HuffmanCodec<RunlengthToken<zero_bits, data_bits>>& codec)
{
    codec.encode_tokens(RunlengthToken<zero_bits, data_bits>::rle_encode(data), bs);
}


// decode known-size sequence of integers using MPEG like rle+huffman scheme
template<int zero_bits, int data_bits>
std::vector<int> decode_runlength_huffman(
    size_t num_values,
    BitStreamReader& reader, 
    const HuffmanCodec<RunlengthToken<zero_bits, data_bits>>& codec) 
{
    std::vector<int> res;
    while(res.size() < num_values) {
        RunlengthToken<zero_bits, data_bits> rlt = codec.read_next(reader);
        if(rlt.type == rlt.DEFAULT) {
            for(int i = 0; i < rlt.nzeros; i++) {
                res.push_back(0);
            }
            res.push_back(rlt.data);
        } else {
            assert(rlt.type == rlt.END);
            while(res.size() < num_values) 
                res.push_back(0);
        }
    }
    return res;
}

