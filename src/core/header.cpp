#include "header.h"

Header::Header(uint16_t rows, uint16_t cols) : rows(rows), cols(cols) {};

Header::Header(BitStreamReader& reader) {
    uint8_t preamble = reader.read8u();
    if (preamble != this->preamble) {
        std::cout   << "WARNING: Preamble " << preamble 
                    << " did not match expected value (" 
                    << this->preamble << ")" << std::endl;
    }

    rows = reader.read16u();
    cols = reader.read16u();

}

void Header::encode(BitStream& stream) {
    stream.append(preamble, 8);
    stream.append(rows, 16);
    stream.append(cols, 16);
};