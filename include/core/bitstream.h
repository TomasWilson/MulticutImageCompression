#pragma once
#include <vector>
#include <cassert>
#include <limits.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <fstream>

// lightweight (but not super efficient) BitStream class, that helps to encode data with odd bit lengths
struct BitStream {

    std::vector<uint64_t> data;
    size_t head = 0;

    public:

        static const size_t BUFFER_SIZE = sizeof(uint64_t) * CHAR_BIT;
        
        BitStream() {
            data.push_back(0);
        }

        static BitStream from_file(std::ifstream& _if) {
            uint32_t n_entries;
            uint32_t head_32;
            _if.read((char*)&n_entries, sizeof(n_entries));
            _if.read((char*)&head_32, sizeof(head_32));

            BitStream res;
            res.head = head_32;
            res.data.clear();
            res.data.reserve(n_entries);

            for(int i = 0; i < n_entries; i++) {
                uint64_t val;
                _if.read((char*)&val, sizeof(val));
                res.data.push_back(val);
            }

            return res;
        }

        static BitStream from_file(const std::string& file_path) {
            std::ifstream _if(file_path, std::ifstream::binary);
            return from_file(_if);
        }

        void write_to_file(std::ofstream& of) const {
            uint32_t n_entries = data.size();
            uint32_t head_32 = head;
            of.write((char*)&n_entries, sizeof(n_entries));
            of.write((char*)&head_32, sizeof(head_32));
            
            for(uint64_t v : data) {
                of.write((char*)&v, sizeof(v));
            }
        }

        void write_to_file(const std::string& out_path) const {
            std::ofstream out(out_path, std::ofstream::binary);
            write_to_file(out);
        }

        // append bits to the bitstream, by extracting the trailing bits of the given value
        // IMPORTANT:  can only contain one-bits in the `bits` least significant bits!!!!
        template<typename T>
        void append(T t, size_t bits) {
            
            assert( (sizeof(T) * CHAR_BIT == bits) || ((t >> bits) == 0) );
            assert( bits <= sizeof(T) * CHAR_BIT && bits <= BUFFER_SIZE );

            if(head + bits >= BUFFER_SIZE) {
                size_t undershoot = sizeof(t) * CHAR_BIT - bits;
                size_t remaining = BUFFER_SIZE - head;
                size_t overshoot = sizeof(t) * CHAR_BIT - undershoot - remaining;
                data[data.size() - 1] |= ((uint64_t(t) << undershoot) >> (undershoot + overshoot));

                size_t shift = (undershoot + remaining - sizeof(t) * CHAR_BIT + BUFFER_SIZE);
                if (shift >= BUFFER_SIZE) {
                    data.push_back(0);
                }
                else {
                    data.push_back(uint64_t(t) << (undershoot + remaining - sizeof(t) * CHAR_BIT + BUFFER_SIZE));
                }

                head = overshoot;
            }
            else {
                data[data.size() - 1] |= uint64_t(t) << (BUFFER_SIZE - (head + bits));
                head += bits;
            }

        }

        template<typename T>
        T read(size_t index, size_t bits) const {
            assert(bits <= sizeof(T) * CHAR_BIT && bits <= 64);
            
            size_t block_index = index / BUFFER_SIZE;
            assert(block_index < data.size());

            size_t offset = index % BUFFER_SIZE;
            assert(offset < BUFFER_SIZE);

            if (offset + bits > BUFFER_SIZE) {
                size_t remaining = BUFFER_SIZE - offset;
                size_t over = bits - remaining;
                uint64_t res = (data[block_index + 1] >> (BUFFER_SIZE - over));
                return res | (data[block_index] << offset) >> (offset - over);
            } else {
                return (data[block_index] << offset) >> (BUFFER_SIZE - bits);
            }
        }

        void pad_to_bytes() {
            if(size() % 8 != 0) {
                append(uint8_t(0), 8 - (size() % 8));
            }
        }

        size_t size() const {
            return (data.size() - 1) * sizeof(uint64_t) * CHAR_BIT + head;
        }

        void print(size_t from = 0) const {
            std::cout << "head = " << head << std::endl;
            for(int idx = from; idx < data.size(); idx++) {
                print_at(idx);
            }
        }

        void print_at(size_t idx) const {
            uint64_t d = data[idx];
            for(int i = 63; i >= 0; i--) {

                if(i != 63 && i % 8 == 7) std::cout << " ";

                if((d >> i) & 0b1) {
                    std::cout << "1";
                }
                else {
                    std::cout << "0";
                }
            }
            std::cout << std::endl;
        }

        std::vector<unsigned char> as_uchar(int read_head) const {
            assert(size() % 8 == 0);
            assert(read_head % 8 == 0);
            std::vector<unsigned char> result;
            while(read_head < size()) {
                result.push_back(read<unsigned char>(read_head, 8));
                read_head += 8;
            }
            return result;
        }

        void append_uchar(const std::vector<unsigned char> data) {
            for(unsigned char c : data) {
                append<unsigned char>(c, 8);
            }
        }

        void append_stream(BitStream& other) {
            size_t head = 0;
            size_t remaining = other.size();

            while(remaining > 64) {
                append<uint64_t>(other.read<uint64_t>(head, 64), 64);
                remaining -= 64;
                head += 64;
            }

            if(remaining > 0) {
                append<uint64_t>(other.read<uint64_t>(head, remaining), remaining);
            }
        }

};

// keeps track of the read head for you, so that the bitstream can be read from more conveniently
struct BitStreamReader {

    const BitStream& bs;
    size_t head = 0;

public:
    BitStreamReader(const BitStream& bs) : bs(bs) {

    }
    
    template<typename T>
    T read(size_t bits) {
        T data = bs.read<T>(head, bits);
        head += bits;
        return data;
    }

    bool read_bit() {
        return read<bool>(1);
    }

    bool empty() {
        return head >= bs.size();
    }

    // TODO: probably can optimize this by batching reads
    std::vector<bool> read_bits(size_t bits) {
        std::vector<bool> res;
        res.reserve(bits);
        for(size_t i = 0; i < bits; i++) {
            res.push_back(read<bool>(1));
        }
        return res;
    }

    uint8_t read8u(size_t bits = 8) {
        return read<uint8_t>(bits);
    }

    uint16_t read16u(size_t bits = 16) {
        return read<uint16_t>(bits);
    }

    uint32_t read32u(size_t bits = 32) {
        return read<uint32_t>(bits);
    }

    // TODO: return a view into a BitStream instead
    BitStream read_substream(size_t n_bits) {
        BitStream out;
        while(n_bits - out.size() >= 64) {
            out.append<uint64_t>(read<uint64_t>(64), 64);
        }
        if((n_bits - out.size()) > 0) {
            out.append<uint64_t>(read<uint64_t>(n_bits - out.size()), n_bits - out.size());
        }
        return out;
    }

};