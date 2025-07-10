"""
High-performance C++ POS tagger with Python interface.

This module provides a Python wrapper around a C++ implementation of the TnT 
(Trigrams'n'Tags) POS tagger for significant performance improvements.

Usage:
    from src.fastparse.pos_cpp import CppTnTTagger
    
    tagger = CppTnTTagger()
    tagger.train(training_data)
    result = tagger.tag(['The', 'cat', 'sits'])
"""

import time
import random
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import statistics

try:
    # Import the C++ extension module
    from . import fastpos_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ extension not available. Install pybind11 and compile the extension.")

class CppTnTTagger:
    """
    Python wrapper for the C++ TnT POS tagger implementation.
    
    Provides the same interface as the Python TnT tagger but with significant
    performance improvements through C++ implementation.
    """
    
    def __init__(self):
        """Initialize the C++ TnT tagger."""
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ extension not available. Please compile the fastpos_cpp module.")
        
        self._cpp_tagger = fastpos_cpp.TnTTagger()
        self._trained = False
        
    def train(self, training_data):
        """
        Train the tagger on tagged sentences.
        
        Args:
            training_data: List of sentences, where each sentence is a list of 
                          (word, tag) tuples. Compatible with NLTK corpus format.
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Convert Python data to C++ format if needed
        if hasattr(training_data[0][0], '__len__') and len(training_data[0][0]) == 2:
            # Already in the right format: list of (word, tag) tuples
            cpp_training_data = fastpos_cpp.convert_nltk_corpus(training_data)
        else:
            # Convert if needed
            cpp_training_data = []
            for sentence in training_data:
                cpp_sentence = [(word, tag) for word, tag in sentence]
                cpp_training_data.append(cpp_sentence)
        
        self._cpp_tagger.train(cpp_training_data)
        self._trained = True
        
    def tag(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a sentence.
        
        Args:
            words: List of words to tag
            
        Returns:
            List of (word, tag) tuples
        """
        if not self._trained:
            raise RuntimeError("Tagger must be trained before tagging")
        
        return self._cpp_tagger.tag(words)
    
    def save_model(self, filename: str):
        """Save the trained model to a file."""
        if not self._trained:
            raise RuntimeError("Cannot save untrained model")
        self._cpp_tagger.save_model(filename)
    
    def load_model(self, filename: str):
        """Load a trained model from a file."""
        self._cpp_tagger.load_model(filename)
        self._trained = True
    
    @property
    def vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return self._cpp_tagger.vocabulary_size()
    
    @property
    def tag_count(self) -> int:
        """Get the number of unique POS tags."""
        return self._cpp_tagger.tag_count()
    
    @property
    def training_time(self) -> float:
        """Get the training time in seconds."""
        return self._cpp_tagger.training_time()
    
    def __repr__(self):
        if self._trained:
            return f"<CppTnTTagger: {self.vocabulary_size:,} words, {self.tag_count} tags>"
        else:
            return "<CppTnTTagger: untrained>"

def train_and_benchmark_cpp_tagger(dataset_name='brown', max_sentences=None):
    """
    Train and benchmark the C++ TnT tagger on a dataset.
    
    Args:
        dataset_name: 'brown', 'treebank', or 'combined'
        max_sentences: Maximum number of sentences to use
        
    Returns:
        Dictionary with performance metrics and trained tagger
    """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ extension not available")
    
    # Import the data loading function from the Python version
    from .pos import load_large_dataset, evaluate_accuracy
    
    print("=" * 60)
    print(f"TRAINING C++ TnT POS TAGGER ON {dataset_name.upper()} DATASET")
    print("=" * 60)
    
    # Load dataset
    start_time = time.time()
    train_data, test_data = load_large_dataset(dataset_name, max_sentences)
    load_time = time.time() - start_time
    print(f"Dataset loading time: {load_time:.2f} seconds")
    
    # Calculate dataset statistics
    total_tokens = sum(len(sent) for sent in train_data + test_data)
    unique_words = set()
    unique_tags = set()
    for sent in train_data + test_data:
        for word, tag in sent:
            unique_words.add(word.lower())
            unique_tags.add(tag)
    
    print(f"Dataset statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique words: {len(unique_words):,}")
    print(f"  Unique POS tags: {len(unique_tags)}")
    
    # Train the C++ tagger
    print("\nTraining C++ TnT tagger...")
    tagger = CppTnTTagger()
    
    training_start = time.time()
    tagger.train(train_data)
    training_time = time.time() - training_start
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training speed: {len(train_data) / training_time:.1f} sentences/second")
    print(f"Training speed: {sum(len(s) for s in train_data) / training_time:.1f} tokens/second")
    
    # Benchmark inference speed using C++ benchmark function
    print("\nBenchmarking C++ inference speed...")
    test_sentences_words = [[word for word, tag in sent] for sent in test_data[:100]]
    
    cpp_benchmark_results = fastpos_cpp.benchmark_tagging(
        tagger._cpp_tagger, test_sentences_words, iterations=100
    )
    
    print(f"C++ Benchmark Results:")
    print(f"  {cpp_benchmark_results['tokens_per_second']:.1f} tokens/second")
    print(f"  {cpp_benchmark_results['sentences_per_second']:.1f} sentences/second")
    print(f"  {cpp_benchmark_results['avg_time_per_sentence']*1000:.3f} ms/sentence")
    
    # Evaluate accuracy (convert to Python format for evaluation)
    print("\nEvaluating accuracy...")
    correct_tokens = 0
    total_tokens = 0
    
    for sent in test_data[:200]:  # Use subset for faster evaluation
        words = [word for word, tag in sent]
        true_tags = [tag for word, tag in sent]
        predicted = tagger.tag(words)
        pred_tags = [tag for word, tag in predicted]
        
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            total_tokens += 1
            if true_tag == pred_tag:
                correct_tokens += 1
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    results = {
        'dataset': dataset_name,
        'dataset_stats': {
            'total_sentences': len(train_data) + len(test_data),
            'train_sentences': len(train_data),
            'test_sentences': len(test_data),
            'total_tokens': total_tokens,
            'unique_words': len(unique_words),
            'unique_tags': len(unique_tags)
        },
        'training': {
            'time_seconds': training_time,
            'sentences_per_second': len(train_data) / training_time,
            'tokens_per_second': sum(len(s) for s in train_data) / training_time
        },
        'accuracy': {
            'token_accuracy': accuracy,
            'correct_tokens': correct_tokens,
            'total_tokens': total_tokens
        },
        'inference_speed': cpp_benchmark_results,
        'cpp_stats': {
            'vocabulary_size': tagger.vocabulary_size,
            'tag_count': tagger.tag_count,
            'training_time': tagger.training_time
        }
    }
    
    print_cpp_benchmark_summary(results)
    
    return results, tagger

def compare_python_vs_cpp():
    """
    Compare performance between Python and C++ implementations.
    """
    if not CPP_AVAILABLE:
        print("C++ extension not available for comparison")
        return
    
    from .pos import load_large_dataset, create_tnt_tagger
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: PYTHON TnT vs C++ TnT")
    print("=" * 80)
    
    # Load test data
    train_data, test_data = load_large_dataset('brown', max_sentences=500)
    test_sentences = test_data[:50]
    
    # Train Python tagger
    print("\nTraining Python TnT tagger...")
    python_start = time.time()
    python_tagger = create_tnt_tagger(train_data)
    python_training_time = time.time() - python_start
    
    # Train C++ tagger
    print("Training C++ TnT tagger...")
    cpp_start = time.time()
    cpp_tagger = CppTnTTagger()
    cpp_tagger.train(train_data)
    cpp_training_time = time.time() - cpp_start
    
    # Benchmark inference - Python
    print("\nBenchmarking Python inference...")
    test_words = [[word for word, tag in sent] for sent in test_sentences]
    
    python_times = []
    for _ in range(50):
        start = time.time()
        for words in test_words:
            python_tagger.tag(words)
        end = time.time()
        python_times.append(end - start)
    
    python_avg_time = sum(python_times) / len(python_times)
    python_tokens_per_sec = sum(len(words) for words in test_words) * 50 / sum(python_times)
    
    # Benchmark inference - C++
    print("Benchmarking C++ inference...")
    cpp_benchmark = fastpos_cpp.benchmark_tagging(cpp_tagger._cpp_tagger, test_words, 50)
    
    # Print comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Python TnT':<20} {'C++ TnT':<20} {'Speedup':<15}")
    print("-" * 85)
    print(f"{'Training time (s)':<30} {python_training_time:<20.3f} {cpp_training_time:<20.3f} {python_training_time/cpp_training_time:<15.1f}x")
    print(f"{'Tokens/second':<30} {python_tokens_per_sec:<20.1f} {cpp_benchmark['tokens_per_second']:<20.1f} {cpp_benchmark['tokens_per_second']/python_tokens_per_sec:<15.1f}x")
    print(f"{'Vocab size':<30} {'N/A':<20} {cpp_tagger.vocabulary_size:<20,} {'-':<15}")
    print(f"{'Tag count':<30} {'N/A':<20} {cpp_tagger.tag_count:<20} {'-':<15}")
    
    return {
        'python_training_time': python_training_time,
        'cpp_training_time': cpp_training_time,
        'python_tokens_per_sec': python_tokens_per_sec,
        'cpp_tokens_per_sec': cpp_benchmark['tokens_per_second'],
        'training_speedup': python_training_time / cpp_training_time,
        'inference_speedup': cpp_benchmark['tokens_per_second'] / python_tokens_per_sec
    }

def print_cpp_benchmark_summary(results):
    """Print a formatted summary of C++ benchmark results."""
    print("\n" + "=" * 60)
    print("C++ TnT BENCHMARK SUMMARY")
    print("=" * 60)
    
    ds = results['dataset_stats']
    tr = results['training']
    acc = results['accuracy']
    speed = results['inference_speed']
    cpp = results['cpp_stats']
    
    print(f"Dataset: {results['dataset']}")
    print(f"  {ds['total_sentences']:,} sentences ({ds['train_sentences']:,} train, {ds['test_sentences']:,} test)")
    print(f"  {ds['total_tokens']:,} tokens, {ds['unique_words']:,} unique words, {ds['unique_tags']} POS tags")
    
    print(f"\nTraining Performance:")
    print(f"  Time: {tr['time_seconds']:.2f} seconds")
    print(f"  Speed: {tr['sentences_per_second']:.1f} sentences/sec, {tr['tokens_per_second']:.1f} tokens/sec")
    
    print(f"\nAccuracy:")
    print(f"  Token accuracy: {acc['token_accuracy']:.1%} ({acc['correct_tokens']:,}/{acc['total_tokens']:,})")
    
    print(f"\nInference Speed (C++):")
    print(f"  {speed['tokens_per_second']:,.1f} tokens/second")
    print(f"  {speed['sentences_per_second']:,.1f} sentences/second")
    print(f"  {speed['avg_time_per_sentence']*1000:.3f} ms/sentence")
    
    print(f"\nC++ Tagger Statistics:")
    print(f"  Vocabulary: {cpp['vocabulary_size']:,} words")
    print(f"  Tags: {cpp['tag_count']} unique tags")
    print(f"  C++ training time: {cpp['training_time']:.3f} seconds")

def demo_cpp_tagger():
    """
    Demonstrate the C++ POS tagger with sample text.
    """
    if not CPP_AVAILABLE:
        print("C++ extension not available for demo")
        return
    
    print("C++ TnT Tagger Demo")
    print("=" * 50)
    
    # Create sample training data
    sample_corpus = [
        [('The', 'DT'), ('cat', 'NN'), ('sits', 'VBZ'), ('on', 'IN'), ('the', 'DT'), ('mat', 'NN')],
        [('Dogs', 'NNS'), ('bark', 'VBP'), ('loudly', 'RB')],
        [('She', 'PRP'), ('runs', 'VBZ'), ('quickly', 'RB')],
        [('A', 'DT'), ('big', 'JJ'), ('dog', 'NN'), ('sleeps', 'VBZ')],
        [('Birds', 'NNS'), ('fly', 'VBP'), ('high', 'RB')]
    ]
    
    # Train tagger
    print("Training C++ tagger on sample data...")
    tagger = CppTnTTagger()
    tagger.train(sample_corpus)
    
    print(f"Training completed: {tagger}")
    print(f"Training time: {tagger.training_time:.4f} seconds")
    
    # Test on new sentences
    test_sentences = [
        ['The', 'dog', 'runs'],
        ['Cats', 'sleep', 'peacefully'],
        ['A', 'small', 'bird', 'flies']
    ]
    
    print("\nTagging test sentences:")
    for sentence in test_sentences:
        tagged = tagger.tag(sentence)
        print(f"  {sentence} → {tagged}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_cpp_tagger()
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_python_vs_cpp()
    elif len(sys.argv) > 1 and sys.argv[1] == 'small':
        train_and_benchmark_cpp_tagger('brown', max_sentences=1000)
    else:
        print("C++ TnT POS Tagger")
        print("Usage:")
        print("  python pos_cpp.py demo     - Run demo")
        print("  python pos_cpp.py compare  - Compare Python vs C++")
        print("  python pos_cpp.py small    - Quick benchmark")
        
        if CPP_AVAILABLE:
            print("\nC++ extension is available!")
            demo_cpp_tagger()
        else:
            print("\nC++ extension not available. Please compile first.") 