#include "multicut_aware_codec.h"

MulticutAwareCodec MulticutAwareCodec::row_adaptive_col_adaptive = MulticutAwareCodec(
    std::make_unique<AdapativeBitwiseCodecFactory>(4096, 4), 
    std::make_unique<AdapativeBitwiseCodecFactory>(512, 2)
);