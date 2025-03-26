# DisCoCirc Tensor Training Framework

This repository contains a framework for training and evaluating DisCoCirc (Distributional Compositional Categorical) models using tensor-based regression approaches. The framework implements various methods for building and training embeddings for compositional language understanding.

## Overview

DisCoCirc is a framework that combines distributional semantics with categorical compositional models to understand the meaning of language compositionally. This implementation focuses on tensor-based approaches to learn and evaluate compositional embeddings.

## Project Structure

```
.
├── preprocessing/                  # Data preprocessing utilities
│   ├── make_block.py             # Continuous corpora assembly
│   ├── transitive_verb_parser.py # Transitive verb parsing
│   ├── adjective_parse.py        # Adjective-noun pair parsing
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

## Computational Efficiency

The framework implements two main approaches for embedding generation, each with different computational characteristics:

### Non-Contextual Approach (Hybrid)
- Uses a combination of FastText and SentenceTransformer (all-MiniLM-L6-v2)
- Computational complexity: O(n) where n is the number of sentences
- Memory requirements: O(n × 384) for empirical embeddings
- Key operations:
  - FastText embedding lookup: O(1) per word
  - SentenceTransformer encoding: O(1) per sentence
  - PCA dimensionality reduction: O(n × 384²)
- Advantages:
  - Lower memory footprint
  - Faster inference time
  - No need for context window processing

### BERT-Based Approach
- Uses BERT-base-uncased model
- Computational complexity: O(n × l) where n is number of sentences and l is sequence length
- Memory requirements: O(n × 768) for BERT embeddings
- Key operations:
  - Tokenization: O(l) per sentence
  - BERT forward pass: O(l²) per sentence
  - Hidden state extraction: O(1) per sentence
- Advantages:
  - Better contextual understanding
  - More robust to polysemy
  - State-of-the-art performance

### Comparison
- Training Time: Non-contextual approach is typically 2-3x faster
- Memory Usage: Non-contextual approach uses ~50% less memory
- Inference Speed: Non-contextual approach is 3-4x faster
- Quality: BERT approach generally provides better semantic understanding but at higher computational cost

## Dependencies

- PyTorch
- NumPy
- BERT (for BERT-based embeddings)

## Testing

The project includes a comprehensive test suite in the `tests/` directory to validate the functionality of various components.


