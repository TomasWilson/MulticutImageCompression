#pragma once
#include "bitstream.h"

struct Header {

    uint8_t preamble = 0xFF;
    uint16_t rows, cols;

    Header() = default;
    Header(uint16_t rows, uint16_t cols);
    Header(BitStreamReader& reader);

    void encode(BitStream& stream);
};