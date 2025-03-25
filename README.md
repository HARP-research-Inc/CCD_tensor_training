# DisCoCirc Tensor Training Framework

This repository contains a framework for training and evaluating DisCoCirc (Distributional Compositional Categorical) models using tensor-based regression approaches. The framework implements various methods for building and training embeddings for compositional language understanding.

## Overview

DisCoCirc is a framework that combines distributional semantics with categorical compositional models to understand the meaning of language compositionally. This implementation focuses on tensor-based approaches to learn and evaluate compositional embeddings.

## Project Structure

```
.
├── preprocessing/           # Data preprocessing utilities
│   ├── make_block.py       # Block matrix construction
│   ├── smart_parser.py     # Advanced parsing utilities
│   ├── adjective_parse.py  # Adjective-specific parsing
│   └── build_SVO_sentences.py  # Subject-Verb-Object sentence construction
├── tests/                  # Test suite
├── documents/             # Documentation and examples
├── util.py               # Utility functions and shared components
├── full_rank_regression.py    # Full-rank tensor regression implementation
├── build_adj_embeddings.py   # Adjective embedding construction
├── BERT_build_embeddings.py  # BERT-based embedding generation
└── transitive_non_contxtl_build_embeddings.py  # Non-contextual embedding construction
```

## Key Components

### Preprocessing
- `preprocessing/`: Contains utilities for parsing and preparing data
  - `smart_parser.py`: Handles complex parsing scenarios
  - `adjective_parse.py`: Specialized parsing for adjective constructions
  - `build_SVO_sentences.py`: Constructs Subject-Verb-Object sentence structures

### Core Training
- `full_rank_regression.py`: Implements full-rank tensor regression for compositional learning
- `build_adj_embeddings.py`: Handles adjective embedding construction
- `BERT_build_embeddings.py`: Generates embeddings using BERT models
- `transitive_non_contxtl_build_embeddings.py`: Creates non-contextual embeddings

### Utilities
- `util.py`: Contains shared utility functions and components
- `load_in_parallel.py`: Parallel data loading utilities

## Usage

The framework supports various training scenarios:

1. Two-word regression for paired words
2. Three-word regression for more complex compositions
3. Adjective embedding construction
4. BERT-based embedding generation

## Dependencies

- PyTorch
- NumPy
- BERT (for BERT-based embeddings)

## Testing

The project includes a comprehensive test suite in the `tests/` directory to validate the functionality of various components.


