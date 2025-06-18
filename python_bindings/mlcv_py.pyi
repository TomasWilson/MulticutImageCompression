import numpy as np
from typing import Tuple

def huffman_mean_grid(img: np.ndarray, compression_strength: float = 1, cell_size: int = 128) -> Tuple[np.ndarray, int]:
    ...


def optimize_grid_mean(img: np.ndarray, compression_strength: float = 1, cell_size: int = 128) -> np.ndarray:
    ...


def test_huffman_encoding(mask: np.ndarray) -> int:
    ...

def test_adaptive_multicut_aware_encoding(
    mask: np.ndarray, 
    row_context_size: int = 4096,
    row_order: int = 4,
    col_context_size: int = 512,
    col_order: int = 2
) -> int:
    ...

def test_border_encoding(mask: np.ndarray) -> int:
    ...

def test_ensemble_encoding(mask: np.ndarray, optimization_level: float) -> int:
    ...

def make_mask_with_size(img, multicut_codec, partition_codec, optimizer, compression_strength) -> Tuple[np.ndarray, int]:
    ...

def encode_mask_with_size(img, mask, multicut_codec, partition_codec, entropy_compress, optim_level: float=0) -> Tuple[np.ndarray, int]:
    ...

from enum import Enum

class MULTICUT_CODEC(Enum):
    HUFFMAN = 0
    BORDER = 1
    MULTICUT_AWARE = 2
    ENSEMBLE = 3

class PARTITION_CODEC(Enum):
    SIMPLE = 0
    DIFFERENTIAL = 1

class OPTIMIZER(Enum):
    LOSSLESS = 0
    GREEDY = 1
    GREEDY_GRID = 2