#pragma once
#include "bitstream.h"

#include "BitIoStream.hpp"
#include "FrequencyTable.hpp"
#include "ArithmeticCoder.hpp"

#include <iostream>
#include <sstream>

// this file features two utility classes, that provide an adapter from
// the BitInputStream and BitOutputStream classes in the arithmetic coder library,
// to my own BitStream implementation. 

class BitInputStreamAdapter : public BitInputStream {

    BitStreamReader& reader;
    static inline std::istream proxy_stream = std::istream(nullptr);

public:

    BitInputStreamAdapter(BitStreamReader& reader) : BitInputStream(proxy_stream), reader(reader)
    {

    }

    int read() {
        if(!reader.empty()) {
            return reader.read_bit(); 
        }
        return -1;
    }

    int readNoEof() {
        int res = read();
        if(res == -1) {
		    throw std::runtime_error("End of stream");
        }
        return res;
    }

};

class BitOutputStreamAdapter : public BitOutputStream {

    BitStream& bs;
    static inline std::ostream proxy_stream = std::ostream(nullptr);

public:

    BitOutputStreamAdapter(BitStream& bs) : BitOutputStream(proxy_stream), bs(bs) {

    }

    void write(int b) {
        bs.append<bool>(b, 1);
    }

    void finish() {
        bs.pad_to_bytes();
    }

};

class WrappedArithmeticEncoder {

    BitStream bs;
    BitOutputStreamAdapter adapter;
    ArithmeticEncoder enc;

public:

    WrappedArithmeticEncoder() : adapter(bs), enc(32, adapter) {

    }

    void write(const FrequencyTable &freqs, std::uint32_t symbol) {
        enc.write(freqs, symbol);
    }

    void finish(BitStream& out) {
        enc.finish();
        out.append<uint32_t>(bs.size(), 32);
        out.append_stream(bs);
    }

};

class WrappedArithmeticDecoder {

    BitStream data;
    BitStreamReader data_reader;

    BitInputStreamAdapter adapter;
    ArithmeticDecoder dec;

public:

    WrappedArithmeticDecoder(BitStreamReader& reader) : 
        data_reader(init_reader(reader)), 
        adapter(data_reader), 
        dec(32, adapter) {

    }

    uint32_t read(const FrequencyTable &freqs) {
        return dec.read(freqs);
    }

private:
    
    BitStreamReader init_reader(BitStreamReader& reader) {
        uint32_t n_bits = reader.read32u();
        data = reader.read_substream(n_bits);
        return BitStreamReader(data);
    }

};

template<int vmin, int vmax>
void encode_sequence(std::vector<int> data, BitStream& bs, uint32_t token_freq_bits) {
    static_assert(vmin < vmax);

    std::vector<uint32_t> token_freqs(vmax - vmin + 1, 0);
    for(int i : data) {
        token_freqs[i - vmin]++;
    }

    uint32_t max_freq = *std::max_element(token_freqs.begin(), token_freqs.end());
    uint32_t max_encode_freq = (0b1 << token_freq_bits) - 1;

    for(uint32_t& f : token_freqs) {
        if(f > 0) {
            f = std::clamp(
                uint32_t(double(f) / double(max_freq) * double(max_encode_freq)),
                uint32_t(1),
                max_encode_freq
            );
        }

        bs.append<uint32_t>(f, token_freq_bits);
    }

    bs.append<uint32_t>(data.size(), 32);

    WrappedArithmeticEncoder encoder;
    auto ftable = SimpleFrequencyTable(token_freqs);
    for(int i : data) {
        encoder.write(ftable, i - vmin);
    }
    encoder.finish(bs);
}

template<int vmin, int vmax>
std::vector<int> decode_sequence(BitStreamReader& reader, uint32_t token_freq_bits) {
    static_assert(vmin < vmax);

    std::vector<uint32_t> token_freqs;
    for(int i = vmin; i <= vmax; i++) {
        token_freqs.push_back(reader.read<uint32_t>(token_freq_bits));
    }

    uint32_t ntokens = reader.read32u();
    std::vector<int> res;
    res.reserve(ntokens);

    auto ftable = SimpleFrequencyTable(token_freqs);
    WrappedArithmeticDecoder decoder(reader);

    for(int i = 0; i < ntokens; i++) {
        res.push_back(decoder.read(ftable) + vmin);
    }

    return res;
}

// template<int vmin, int vmax>
// SimpleFrequencyTable build_and_encode_table(std::vector<int> data, uint32_t token_freq_bits, BitStream& bs) {
//     static_assert(vmin < vmax);

//     std::unordered_map<int, uint32_t> token_freqs;
//     for(int i : data) {
//         token_freqs[i]++;
//     }

//     uint32_t max_freq = 0;
//     for(const auto&[token, freq] : token_freqs) {
//         if (freq > max_freq) max_freq = freq;
//     }

//     uint32_t max_encode_freq = 0b1 << token_freq_bits - 1;

//     std::vector<uint32_t> encode_freqs;
//     encode_freqs.reserve(vmax - vmin + 1);

//     for(int i = vmin; i <= vmax; i++) {
//         if(token_freqs.find(i) == token_freqs.end()) {
//             encode_freqs.push_back(0);
//             bs.append<uint32_t>(0, token_freq_bits);
//         }
//         else {
//             uint32_t true_freq = token_freqs.at(i);
//             uint32_t encode_freq = std::clamp(
//                 uint32_t(double(true_freq) / max_freq * max_encode_freq),
//                 uint32_t(1),
//                 max_encode_freq
//             );
//             encode_freqs.push_back(encode_freq);
//             bs.append<uint32_t>(encode_freq, token_freq_bits);
//         }
//     }

//     return SimpleFrequencyTable(encode_freqs);
// }

// template<int vmin, int vmax>
// SimpleFrequencyTable read_and_decode_table(uint32_t token_freq_bits, BitStreamReader& reader) {
//     static_assert(vmin < vmax);

//     std::vector<uint32_t> freqs;
//     freqs.reserve(vmax - vmin + 1);
//     for(int i = vmin; i <= vmax; i++) {
//         uint32_t decode_freq = reader.read<uint32_t>(token_freq_bits);
//         freqs.push_back(decode_freq);
//     }
//     return SimpleFrequencyTable(freqs);
// }

// template<int vmin>
// void encode_sequence(std::vector<int> symbols, SimpleFrequencyTable& freqs, ArithmeticEncoder& encoder) {
//     for(size_t i = 0; i < symbols.size(); i++) {
//         encoder.write(freqs, symbols[i] - vmin);
//     }
// }

// template<int vmin>
// std::vector<int> decode_sequence(uint32_t n, SimpleFrequencyTable& freqs, ArithmeticDecoder& decoder) {
//     std::vector<int> res;
//     res.reserve(n);
//     for(size_t i = 0; i < n; i++) {
//         res.push_back(decoder.read(freqs) + vmin);
//     }
//     return res;
// }
