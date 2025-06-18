#pragma once
#include "bitstream.h"
#include "unordered_map"
#include "map"
#include "arithmetic.h"
#include "ArithmeticCoder.hpp"
#include "FrequencyTable.hpp"
#include "diagnostics.h"

const int DEFAULT_WEIGHT = 10;

class ContextBasedEncoder {
protected:
    BitStream& bs;

public:
    ContextBasedEncoder(BitStream& bs) : bs(bs) {}

    virtual void initialize() {}
    virtual void encode_bit(
        bool data,
        const std::vector<bool>& context = {}
    ) = 0;
    virtual void finalize() {}

    virtual ~ContextBasedEncoder() = default;
};

class ContextBasedDecoder {
protected:
    BitStreamReader& reader;

public:
    ContextBasedDecoder(BitStreamReader& reader) : reader(reader) {}

    virtual void initialize() {}
    virtual bool decode_bit(const std::vector<bool>& context = {}) = 0;
    virtual void finalize() {}

    virtual ~ContextBasedDecoder() = default;
};

class NaiveEncoder : public ContextBasedEncoder {

    using ContextBasedEncoder::ContextBasedEncoder;

public:
    void encode_bit(bool data, const std::vector<bool>& context = {}) {
        bs.append<bool>(data, 1);
    }

};

class NaiveDecoder : public ContextBasedDecoder {
    
    using ContextBasedDecoder::ContextBasedDecoder;

public:
    bool decode_bit(const std::vector<bool>& context = {}) {
        return reader.read_bit();
    }

};

class ArithmeticContextBasedEncoder : public ContextBasedEncoder {

    BitStream sub_stream;
    BitOutputStreamAdapter adapter;

protected:
    ArithmeticEncoder encoder;

public:

    ArithmeticContextBasedEncoder(BitStream& bs) :
        ContextBasedEncoder(bs), 
        adapter(sub_stream),
        encoder(32, adapter) {

    }

    // After calling this (from a child class), the encoding has concluded,
    // and all data will be appended to the output stream. Any further accesses 
    // to the encoder will not be reflected in the output stream (ContextBasedEncoder::bs)
    void finalize() {
        encoder.finish();
        sub_stream.append<uint64_t>(0, 32);
        bs.append<uint32_t>(sub_stream.size(), 32);
        bs.append_stream(sub_stream);
    }

};

class ArithmeticContextBasedDecoder : public ContextBasedDecoder {

    BitStream sub_stream;
    BitInputStreamAdapter adapter;
    size_t head_after_read;

protected:
    std::unique_ptr<ArithmeticDecoder> decoder;

public:

    ArithmeticContextBasedDecoder(BitStreamReader& reader) :
        ContextBasedDecoder(reader), 
        adapter(reader) {

    }

    void initialize() {
        uint32_t read_bits = reader.read32u();
        head_after_read = reader.head + read_bits;
        decoder = std::make_unique<ArithmeticDecoder>(32, adapter);
    }

    void finalize() {
        reader.head = head_after_read;
    }

};

class BlockEncoder : public ArithmeticContextBasedEncoder {

    size_t block_size; // bits per block
    size_t freq_precision; // bits used to encode freqs

    std::vector<bool> current_symbol;
    std::vector<uint64_t> symbols;

public:

    BlockEncoder(BitStream& bs, size_t block_size, size_t freq_precision) 
        : ArithmeticContextBasedEncoder(bs), block_size(block_size), freq_precision(freq_precision) {

        }

    void initialize() {

    }

    void encode_bit(bool data, const std::vector<bool>& context = {}) {
        current_symbol.push_back(data);
        if(current_symbol.size() == block_size) {
            uint64_t packed_symbol = 0;
            for(int b = 0; b < block_size; b++) {
                packed_symbol |= (current_symbol[b] << b);
            }
            symbols.push_back(packed_symbol);
            current_symbol = {};
        }
    }

    void finalize () {

        while(current_symbol.size() != 0) {
            encode_bit(0, {});
        }

        std::unordered_map<uint64_t, uint32_t> symbol_counts;
        for(auto& symbol : symbols) {
            symbol_counts[symbol]++;
        }

        uint64_t max_freq = 0;
        for(const auto& [sym, freq] : symbol_counts) {
            if(freq > max_freq) max_freq = freq;
        }

        uint64_t max_count = 0b1 << freq_precision - 1;
        uint64_t n_symbols = 0b1 << block_size;

        std::vector<uint32_t> encode_counts;
        encode_counts.reserve(n_symbols);

        for(uint64_t i = 0; i < n_symbols; i++) {
            if(symbol_counts.find(i) != symbol_counts.end()) {
                uint64_t true_count = symbol_counts.at(i);
                uint64_t encode_count = std::clamp(
                    uint64_t(double(true_count) / max_freq * max_count),
                    uint64_t(1), // don't let freqs go to zero
                    max_count
                );
                encode_counts.push_back(encode_count);
                bs.append<uint64_t>(encode_count, freq_precision);
            }
            else {
                encode_counts.push_back(0);
                bs.append<uint64_t>(0, freq_precision);
            }
        }

        SimpleFrequencyTable freqs(encode_counts);

        for(int i = 0; i < symbols.size(); i++) {
            encoder.write(freqs, symbols[i]);
        }

        ArithmeticContextBasedEncoder::finalize();
    }

};

class BlockDecoder : public ArithmeticContextBasedDecoder {

    size_t block_size;
    size_t freq_precision;
    
    SimpleFrequencyTable freqs;
    std::vector<bool> current_symbol;

public:
    BlockDecoder(BitStreamReader& reader, size_t block_size, size_t freq_precision) 
        : ArithmeticContextBasedDecoder(reader), 
        block_size(block_size), 
        freq_precision(freq_precision), 
        freqs(std::vector<uint32_t>(0b1 << block_size)) {

    }

    void initialize() {
        uint64_t n_symbols = 0b1 << block_size;
        for(int i = 0; i < n_symbols; i++) {
            freqs.set(i, reader.read32u(freq_precision));
        }

        ArithmeticContextBasedDecoder::initialize();
    }

    bool decode_bit(const std::vector<bool>& context = {}) {
        if(current_symbol.empty()) {
            uint64_t packed_symbol = decoder->read(freqs);
            for(int b = block_size - 1; b >= 0; b--) {
                bool data = (packed_symbol >> b) & 0b1;
                current_symbol.push_back(data);
            }
        }
        bool res = current_symbol.back();
        current_symbol.pop_back();
        return res;
    }

    void finalize() {
        ArithmeticContextBasedDecoder::finalize();
    }

};

struct SlidingWindowHistory {

    size_t max_window_size;
    std::deque<bool> window;
    size_t context_size;

    std::vector<uint32_t[2]> freqs;
    uint32_t context_mask;

public: 

    size_t current_context = 0;
    size_t last_context = 0;

    SlidingWindowHistory(size_t max_window_size, size_t context_size) : 
        max_window_size(max_window_size),
        context_size(context_size),
        freqs(0b1 << context_size),
        context_mask((0b1 << context_size) - 1) {

    }

    void add(bool b) {

        if(window.size() >= context_size) {
            freqs[current_context][b]++;
        }

        if(window.size() == max_window_size) {
            bool new_delete_bit = window.at(max_window_size - context_size - 1);
            freqs[last_context][new_delete_bit]--;
            last_context = ((last_context << 1) & context_mask) | new_delete_bit;
            window.pop_back();
        }
        else if(window.size() >= (max_window_size - context_size)) {
            last_context = (last_context << 1) | window.at(max_window_size - context_size - 1);
        }

        current_context = ((current_context << 1) & context_mask) | b;
        window.push_front(b);

    }

    std::vector<uint32_t> get_current_context_freqs() {
        uint32_t count_0 = freqs[current_context][0];
        uint32_t count_1 = freqs[current_context][1];
        return {count_0, count_1};
    }

};

class AdaptiveBitwiseEncoder : public ArithmeticContextBasedEncoder {

    // how many bits in the past to consider
    // current probability distribution will be estimated from these bits
    // can be set to large number (i.e. 2^63)  n   to consider all previous transitions
    size_t window_size;

    // how many preceeding symbols to consider
    // i.e. order 0 -> consider only bit freuencies p(0) p(1)
    // order 1 -> consider p(0|0), p(0|1), p(1|0), p(1|1)
    // order 2 -> consider p(0|00), ... p(1|11)
    size_t order;
    SlidingWindowHistory window;

public:

    AdaptiveBitwiseEncoder(BitStream& bs, size_t window_size, size_t order) :
        ArithmeticContextBasedEncoder(bs), 
        window_size(window_size), 
        order(order), 
        window(window_size, order) {

        }

    void initialize() {

    }

    void encode_bit(bool data, const std::vector<bool>& context = {}) {

        auto freqs = window.get_current_context_freqs();

        if(freqs[0] == 0 && freqs[1] == 0) {
            freqs[0] = 1;
            freqs[1] = 1;
        }
        else if(freqs[0] == 0) {
            freqs[0] = 1;
            freqs[1] = DEFAULT_WEIGHT;
        }
        else if(freqs[1] == 0) {
            freqs[0] = DEFAULT_WEIGHT;
            freqs[1] = 1;
        }

        SimpleFrequencyTable f({freqs[0], freqs[1]});
        encoder.write(f, data);
        window.add(data);

    }

    void finalize() {
        ArithmeticContextBasedEncoder::finalize();
    }

};

class AdaptiveBitwiseDecoder : public ArithmeticContextBasedDecoder {

    // see AdaptiveBitwiseEncoder
    size_t window_size;

    size_t order;
    std::vector<uint32_t> symbol_counts;
    SlidingWindowHistory window;

    int ctr = 0;

public:

    AdaptiveBitwiseDecoder(BitStreamReader& reader, size_t window_size, size_t order) :
        ArithmeticContextBasedDecoder(reader), 
        window_size(window_size), 
        order(order), 
        window(window_size, order) {

    }

    void initialize() {
        ArithmeticContextBasedDecoder::initialize();
    }

    bool decode_bit(const std::vector<bool>& context = {}) {

        auto freqs = window.get_current_context_freqs();

        if(freqs[0] == 0 && freqs[1] == 0) {
            freqs[0] = 1;
            freqs[1] = 1;
            ctr++;
        }
        else if(freqs[0] == 0) {
            freqs[0] = 1;
            freqs[1] = DEFAULT_WEIGHT;
            ctr++;
        }
        else if(freqs[1] == 0) {
            freqs[0] = DEFAULT_WEIGHT;
            freqs[1] = 1;
            ctr++;
        }

        SimpleFrequencyTable f({freqs[0], freqs[1]});
        bool data = decoder->read(f);
        window.add(data);
        return data;

    }

    void finalize() {
        ArithmeticContextBasedDecoder::finalize();
    }

};
struct AbstractCodecFactory {
    virtual std::unique_ptr<AbstractCodecFactory> clone() const = 0;
    virtual std::unique_ptr<ContextBasedEncoder> make_encoder(BitStream& bs) const = 0;
    virtual std::unique_ptr<ContextBasedDecoder> make_decoder(BitStreamReader& reader) const = 0;

    virtual ~AbstractCodecFactory() = default;
};

template<typename ConcreteEncoder, typename ConcreteDecoder>
struct ConcreteCodecFactory : public AbstractCodecFactory {
    std::unique_ptr<ContextBasedEncoder> make_encoder(BitStream& bs) const {
        return std::make_unique<ConcreteEncoder>(bs);
    }

    std::unique_ptr<ContextBasedDecoder> make_decoder(BitStreamReader& reader) const {
        return std::make_unique<ConcreteDecoder>(reader);
    }

    std::unique_ptr<AbstractCodecFactory> clone() const {
        return std::make_unique<ConcreteCodecFactory>(*this);
    }
};

using NaiveCodecFactory = ConcreteCodecFactory<NaiveEncoder, NaiveDecoder>;

struct BlockCodecFactory : AbstractCodecFactory {

    size_t block_size, freq_precision;
    
    BlockCodecFactory(size_t block_size, size_t freq_precision) : 
        block_size(block_size), 
        freq_precision(freq_precision) {
            assert(block_size <= 16);
            assert(freq_precision <= 32); 
        }

    std::unique_ptr<ContextBasedEncoder> make_encoder(BitStream& bs) const {
        return std::make_unique<BlockEncoder>(bs, block_size, freq_precision);
    }

    std::unique_ptr<ContextBasedDecoder> make_decoder(BitStreamReader& reader) const {
        return std::make_unique<BlockDecoder>(reader, block_size, freq_precision);
    }

    std::unique_ptr<AbstractCodecFactory> clone() const {
        return std::make_unique<BlockCodecFactory>(*this);
    }

};

struct AdapativeBitwiseCodecFactory : AbstractCodecFactory {

    size_t window_size;
    size_t order;

    AdapativeBitwiseCodecFactory(size_t window_size, size_t order) : 
        window_size(window_size), 
        order(order) {}

    std::unique_ptr<ContextBasedEncoder> make_encoder(BitStream& bs) const {
        return std::make_unique<AdaptiveBitwiseEncoder>(bs, window_size, order);
    }

    std::unique_ptr<ContextBasedDecoder> make_decoder(BitStreamReader& reader) const {
        return std::make_unique<AdaptiveBitwiseDecoder>(reader, window_size, order);
    }

    std::unique_ptr<AbstractCodecFactory> clone() const {
        return std::make_unique<AdapativeBitwiseCodecFactory>(*this);
    }

};