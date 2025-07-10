from nltk.tag import tnt
import nltk
import time
import random
from collections import defaultdict
import statistics

# Download required NLTK data if not present
def safe_nltk_download():
    """Safely download NLTK data with error handling."""
    required_datasets = ['brown', 'punkt', 'averaged_perceptron_tagger', 'treebank']
    
    for dataset in required_datasets:
        try:
            if dataset == 'brown':
                nltk.data.find('corpora/brown')
            elif dataset == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif dataset == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif dataset == 'treebank':
                nltk.data.find('corpora/treebank')
        except (LookupError, Exception) as e:
            print(f"Downloading {dataset}... (Error: {type(e).__name__})")
            try:
                nltk.download(dataset, quiet=True)
            except Exception as download_error:
                print(f"Warning: Could not download {dataset}: {download_error}")
                if dataset == 'treebank':
                    print("Treebank download failed, will skip treebank-related functions")

# Call the safe download function
safe_nltk_download()

def load_large_dataset(dataset_name='brown', max_sentences=None):
    """
    Load a large POS-tagged dataset for training.
    
    Args:
        dataset_name: 'brown', 'treebank', or 'combined'
        max_sentences: Limit number of sentences (None for all)
        
    Returns:
        Tuple of (train_data, test_data) - lists of tagged sentences
    """
    print(f"Loading {dataset_name} dataset...")
    
    try:
        if dataset_name == 'brown':
            from nltk.corpus import brown
            tagged_sents = list(brown.tagged_sents())
        elif dataset_name == 'treebank':
            from nltk.corpus import treebank
            tagged_sents = list(treebank.tagged_sents())
        elif dataset_name == 'combined':
            from nltk.corpus import brown
            tagged_sents = list(brown.tagged_sents())
            try:
                from nltk.corpus import treebank
                tree_sents = list(treebank.tagged_sents())
                tagged_sents.extend(tree_sents)
                print(f"Combined Brown ({len(tagged_sents) - len(tree_sents)}) + Treebank ({len(tree_sents)}) datasets")
            except Exception as e:
                print(f"Warning: Could not load treebank, using only Brown corpus: {e}")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        print("Falling back to Brown corpus...")
        from nltk.corpus import brown
        tagged_sents = list(brown.tagged_sents())
    
    if max_sentences:
        tagged_sents = tagged_sents[:max_sentences]
    
    # Shuffle and split into train/test (80/20)
    random.shuffle(tagged_sents)
    split_point = int(0.8 * len(tagged_sents))
    train_data = tagged_sents[:split_point]
    test_data = tagged_sents[split_point:]
    
    print(f"Loaded {len(tagged_sents)} sentences total")
    print(f"Training set: {len(train_data)} sentences")
    print(f"Test set: {len(test_data)} sentences")
    
    return train_data, test_data

def train_and_benchmark_tagger(dataset_name='brown', max_sentences=None):
    """
    Train TnT tagger on large dataset and benchmark performance.
    
    Args:
        dataset_name: Dataset to use for training
        max_sentences: Limit sentences for faster testing
        
    Returns:
        Dictionary with training time, accuracy metrics, and speed benchmarks
    """
    print("=" * 60)
    print(f"TRAINING TnT POS TAGGER ON {dataset_name.upper()} DATASET")
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
    print(f"  Tags: {sorted(unique_tags)}")
    
    # Train the tagger
    print("\nTraining TnT tagger...")
    tagger = tnt.TnT()
    
    training_start = time.time()
    tagger.train(train_data)
    training_time = time.time() - training_start
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training speed: {len(train_data) / training_time:.1f} sentences/second")
    print(f"Training speed: {sum(len(s) for s in train_data) / training_time:.1f} tokens/second")
    
    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    accuracy_results = evaluate_accuracy(tagger, test_data)
    
    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    speed_results = benchmark_inference_speed(tagger, test_data)
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'dataset_stats': {
            'total_sentences': len(train_data) + len(test_data),
            'train_sentences': len(train_data),
            'test_sentences': len(test_data),
            'total_tokens': total_tokens,
            'unique_words': len(unique_words),
            'unique_tags': len(unique_tags),
            'tags': sorted(unique_tags)
        },
        'training': {
            'time_seconds': training_time,
            'sentences_per_second': len(train_data) / training_time,
            'tokens_per_second': sum(len(s) for s in train_data) / training_time
        },
        'accuracy': accuracy_results,
        'inference_speed': speed_results
    }
    
    # Print summary
    print_benchmark_summary(results)
    
    return results, tagger

def evaluate_accuracy(tagger, test_data):
    """
    Evaluate tagger accuracy on test data.
    
    Args:
        tagger: Trained TnT tagger
        test_data: List of tagged sentences for testing
        
    Returns:
        Dictionary with accuracy metrics
    """
    correct_tokens = 0
    total_tokens = 0
    correct_sentences = 0
    
    tag_confusion = defaultdict(lambda: defaultdict(int))  # actual -> predicted counts
    
    eval_start = time.time()
    
    for sent in test_data:
        # Extract words and true tags
        words = [word for word, tag in sent]
        true_tags = [tag for word, tag in sent]
        
        # Get predictions
        predicted_tags = tagger.tag(words)
        pred_tags_only = [tag for word, tag in predicted_tags]
        
        # Count accuracy
        sent_correct = True
        for true_tag, pred_tag in zip(true_tags, pred_tags_only):
            total_tokens += 1
            tag_confusion[true_tag][pred_tag] += 1
            if true_tag == pred_tag:
                correct_tokens += 1
            else:
                sent_correct = False
        
        if sent_correct:
            correct_sentences += 1
    
    eval_time = time.time() - eval_start
    
    # Calculate metrics
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    sentence_accuracy = correct_sentences / len(test_data) if test_data else 0
    
    # Find most confused tags
    most_confused = []
    for true_tag, predictions in tag_confusion.items():
        total_true = sum(predictions.values())
        for pred_tag, count in predictions.items():
            if true_tag != pred_tag and count > 5:  # Significant confusion
                confusion_rate = count / total_true
                most_confused.append((true_tag, pred_tag, count, confusion_rate))
    
    most_confused.sort(key=lambda x: x[3], reverse=True)  # Sort by confusion rate
    
    return {
        'token_accuracy': token_accuracy,
        'sentence_accuracy': sentence_accuracy,
        'correct_tokens': correct_tokens,
        'total_tokens': total_tokens,
        'correct_sentences': correct_sentences,
        'total_sentences': len(test_data),
        'evaluation_time': eval_time,
        'most_confused_tags': most_confused[:10]  # Top 10 confusions
    }

def benchmark_inference_speed(tagger, test_data, num_iterations=100):
    """
    Benchmark inference speed of the tagger.
    
    Args:
        tagger: Trained TnT tagger
        test_data: Test sentences to use for timing
        num_iterations: Number of timing iterations
        
    Returns:
        Dictionary with speed metrics
    """
    # Select random test sentences for timing
    test_sentences = random.sample(test_data, min(num_iterations, len(test_data)))
    
    sentence_times = []
    token_times = []
    total_tokens = 0
    
    # Warm up
    for _ in range(5):
        if test_sentences:
            words = [word for word, tag in test_sentences[0]]
            tagger.tag(words)
    
    # Time individual sentences
    for sent in test_sentences:
        words = [word for word, tag in sent]
        
        start_time = time.time()
        tagger.tag(words)
        end_time = time.time()
        
        sent_time = end_time - start_time
        sentence_times.append(sent_time)
        token_times.append(sent_time / len(words) if words else 0)
        total_tokens += len(words)
    
    # Calculate bulk processing speed
    all_words = []
    for sent in test_sentences:
        all_words.extend([word for word, tag in sent])
    
    bulk_start = time.time()
    for sent in test_sentences:
        words = [word for word, tag in sent]
        tagger.tag(words)
    bulk_time = time.time() - bulk_start
    
    return {
        'avg_sentence_time': statistics.mean(sentence_times),
        'median_sentence_time': statistics.median(sentence_times),
        'avg_token_time': statistics.mean(token_times),
        'sentences_per_second': len(test_sentences) / bulk_time,
        'tokens_per_second': total_tokens / bulk_time,
        'total_test_tokens': total_tokens,
        'num_test_sentences': len(test_sentences)
    }

def compare_with_nltk_default():
    """
    Compare TnT performance with NLTK's default POS tagger.
    """
    from nltk import pos_tag, word_tokenize
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH NLTK DEFAULT TAGGER")
    print("=" * 60)
    
    # Load small test set
    train_data, test_data = load_large_dataset('brown', max_sentences=1000)
    
    # Train our TnT tagger
    print("Training TnT tagger...")
    tnt_tagger = tnt.TnT()
    tnt_start = time.time()
    tnt_tagger.train(train_data)
    tnt_training_time = time.time() - tnt_start
    
    # Test sentences
    test_sentences = test_data[:100]  # Small set for comparison
    
    # Benchmark TnT
    print("Benchmarking TnT tagger...")
    tnt_correct = 0
    tnt_total = 0
    tnt_start = time.time()
    
    for sent in test_sentences:
        words = [word for word, tag in sent]
        true_tags = [tag for word, tag in sent]
        pred_tags = [tag for word, tag in tnt_tagger.tag(words)]
        
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            tnt_total += 1
            if true_tag == pred_tag:
                tnt_correct += 1
    
    tnt_inference_time = time.time() - tnt_start
    
    # Benchmark NLTK default
    print("Benchmarking NLTK default tagger...")
    nltk_correct = 0
    nltk_total = 0
    nltk_start = time.time()
    
    for sent in test_sentences:
        words = [word for word, tag in sent]
        true_tags = [tag for word, tag in sent]
        pred_tags = [tag for word, tag in pos_tag(words)]
        
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            nltk_total += 1
            if true_tag == pred_tag:
                nltk_correct += 1
    
    nltk_inference_time = time.time() - nltk_start
    
    # Print comparison
    print(f"\nRESULTS COMPARISON:")
    print(f"{'Metric':<25} {'TnT':<15} {'NLTK Default':<15} {'TnT vs NLTK'}")
    print("-" * 65)
    print(f"{'Training time (s)':<25} {tnt_training_time:<15.3f} {'N/A (pre-trained)':<15} {'-'}")
    print(f"{'Accuracy':<25} {tnt_correct/tnt_total:<15.3f} {nltk_correct/nltk_total:<15.3f} {(tnt_correct/tnt_total)/(nltk_correct/nltk_total):.2f}x")
    print(f"{'Inference time (s)':<25} {tnt_inference_time:<15.3f} {nltk_inference_time:<15.3f} {nltk_inference_time/tnt_inference_time:.2f}x faster")
    print(f"{'Tokens/second':<25} {tnt_total/tnt_inference_time:<15.1f} {nltk_total/nltk_inference_time:<15.1f} {'-'}")

def print_benchmark_summary(results):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    ds = results['dataset_stats']
    tr = results['training']
    acc = results['accuracy']
    speed = results['inference_speed']
    
    print(f"Dataset: {results['dataset']}")
    print(f"  {ds['total_sentences']:,} sentences ({ds['train_sentences']:,} train, {ds['test_sentences']:,} test)")
    print(f"  {ds['total_tokens']:,} tokens, {ds['unique_words']:,} unique words, {ds['unique_tags']} POS tags")
    
    print(f"\nTraining Performance:")
    print(f"  Time: {tr['time_seconds']:.2f} seconds")
    print(f"  Speed: {tr['sentences_per_second']:.1f} sentences/sec, {tr['tokens_per_second']:.1f} tokens/sec")
    
    print(f"\nAccuracy:")
    print(f"  Token accuracy: {acc['token_accuracy']:.1%} ({acc['correct_tokens']:,}/{acc['total_tokens']:,})")
    print(f"  Sentence accuracy: {acc['sentence_accuracy']:.1%} ({acc['correct_sentences']:,}/{acc['total_sentences']:,})")
    
    print(f"\nInference Speed:")
    print(f"  {speed['sentences_per_second']:.1f} sentences/second")
    print(f"  {speed['tokens_per_second']:.1f} tokens/second")
    print(f"  {speed['avg_sentence_time']*1000:.2f} ms/sentence (avg)")
    print(f"  {speed['avg_token_time']*1000:.3f} ms/token (avg)")
    
    if acc['most_confused_tags']:
        print(f"\nMost Confused Tag Pairs:")
        for true_tag, pred_tag, count, rate in acc['most_confused_tags'][:5]:
            print(f"  {true_tag} → {pred_tag}: {count} times ({rate:.1%})")

def create_tnt_tagger(training_corpus=None):
    """
    Create and train a TnT POS tagger.
    
    Args:
        training_corpus: List of tagged sentences. If None, uses Brown corpus.
        
    Returns:
        Trained TnT tagger
    """
    tagger = tnt.TnT()
    
    if training_corpus is None:
        # Use Brown corpus as default training data
        from nltk.corpus import brown
        training_corpus = brown.tagged_sents()
    
    tagger.train(training_corpus)
    return tagger

def demo_tagger():
    """
    Demonstrate the POS tagger with sample text.
    """
    # Create sample training data
    sample_corpus = [
        [('The', 'DT'), ('cat', 'NN'), ('sits', 'VBZ'), ('on', 'IN'), ('the', 'DT'), ('mat', 'NN')],
        [('Dogs', 'NNS'), ('bark', 'VBP'), ('loudly', 'RB')],
        [('She', 'PRP'), ('runs', 'VBZ'), ('quickly', 'RB')]
    ]
    
    # Train tagger
    tagger = create_tnt_tagger(sample_corpus)
    
    # Test on new sentences
    test_sentences = [
        ['The', 'dog', 'runs'],
        ['Cats', 'sleep', 'peacefully']
    ]
    
    for sentence in test_sentences:
        tagged = tagger.tag(sentence)
        print(f"Original: {sentence}")
        print(f"Tagged: {tagged}")
        print()

def detailed_performance_analysis():
    """
    Detailed analysis of why TnT is slower than NLTK's default tagger.
    """
    from nltk import pos_tag
    import sys
    
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE ANALYSIS: TnT vs NLTK DEFAULT")
    print("=" * 80)
    
    # Load small test set
    train_data, test_data = load_large_dataset('brown', max_sentences=500)
    test_sentences = test_data[:50]  # Small set for detailed analysis
    
    print(f"\nAnalyzing {len(test_sentences)} test sentences...")
    
    # =============================================
    # 1. WHAT IS NLTK'S DEFAULT TAGGER?
    # =============================================
    print("\n1. NLTK'S DEFAULT TAGGER IDENTIFICATION:")
    print("-" * 50)
    
    # Check what tagger NLTK actually uses
    try:
        import nltk.tag._pos_tag as pos_tag_module
        print(f"NLTK's default tagger: {type(pos_tag_module._POS_TAGGER)}")
        print(f"Tagger class: {pos_tag_module._POS_TAGGER.__class__.__name__}")
    except:
        print("Could not identify NLTK's default tagger class")
    
    # Test a single sentence to see the mechanism
    test_sent = ['The', 'quick', 'brown', 'fox']
    nltk_result = pos_tag(test_sent)
    print(f"NLTK tags for {test_sent}: {nltk_result}")
    
    # =============================================
    # 2. TRAIN TnT TAGGER
    # =============================================
    print("\n2. TRAINING TnT TAGGER:")
    print("-" * 50)
    
    tnt_tagger = tnt.TnT()
    print(f"TnT tagger type: {type(tnt_tagger)}")
    
    training_start = time.time()
    tnt_tagger.train(train_data)
    training_time = time.time() - training_start
    print(f"TnT training time: {training_time:.4f} seconds")
    
    # Check TnT internal structure
    print(f"TnT internal data structures:")
    if hasattr(tnt_tagger, '_emission'):
        print(f"  - Emission probs: {len(tnt_tagger._emission)} entries")
    if hasattr(tnt_tagger, '_transition'):
        print(f"  - Transition probs: {len(tnt_tagger._transition)} entries")
    if hasattr(tnt_tagger, '_tags'):
        print(f"  - Tags: {len(tnt_tagger._tags)} unique tags")
    
    # =============================================
    # 3. DETAILED TIMING BREAKDOWN
    # =============================================
    print("\n3. DETAILED TIMING BREAKDOWN:")
    print("-" * 50)
    
    # Time individual operations
    single_sentence = test_sentences[0]
    words = [word for word, tag in single_sentence]
    
    print(f"Test sentence: {words[:8]}... ({len(words)} tokens)")
    
    # Time TnT operations
    print("\nTnT timing breakdown:")
    
    # Warm up
    for _ in range(3):
        tnt_tagger.tag(words)
    
    # Time single tag operation multiple times
    tnt_times = []
    for _ in range(100):
        start = time.time()
        result = tnt_tagger.tag(words)
        end = time.time()
        tnt_times.append(end - start)
    
    tnt_avg = sum(tnt_times) / len(tnt_times)
    tnt_min = min(tnt_times)
    tnt_max = max(tnt_times)
    
    print(f"  - Average: {tnt_avg*1000:.3f} ms")
    print(f"  - Min:     {tnt_min*1000:.3f} ms") 
    print(f"  - Max:     {tnt_max*1000:.3f} ms")
    print(f"  - Std dev: {(sum((t-tnt_avg)**2 for t in tnt_times)/len(tnt_times))**0.5*1000:.3f} ms")
    
    # Time NLTK operations
    print("\nNLTK timing breakdown:")
    
    # Warm up
    for _ in range(3):
        pos_tag(words)
    
    # Time single tag operation multiple times
    nltk_times = []
    for _ in range(100):
        start = time.time()
        result = pos_tag(words)
        end = time.time()
        nltk_times.append(end - start)
    
    nltk_avg = sum(nltk_times) / len(nltk_times)
    nltk_min = min(nltk_times)
    nltk_max = max(nltk_times)
    
    print(f"  - Average: {nltk_avg*1000:.3f} ms")
    print(f"  - Min:     {nltk_min*1000:.3f} ms")
    print(f"  - Max:     {nltk_max*1000:.3f} ms")
    print(f"  - Std dev: {(sum((t-nltk_avg)**2 for t in nltk_times)/len(nltk_times))**0.5*1000:.3f} ms")
    
    # =============================================
    # 4. MEMORY AND COMPLEXITY ANALYSIS
    # =============================================
    print("\n4. MEMORY AND COMPLEXITY ANALYSIS:")
    print("-" * 50)
    
    # Estimate TnT memory usage
    print("TnT tagger characteristics:")
    
    # Check vocabulary size
    if hasattr(tnt_tagger, '_tokens'):
        vocab_size = len(tnt_tagger._tokens)
        print(f"  - Vocabulary size: {vocab_size:,} words")
    
    # Check tag set size
    if hasattr(tnt_tagger, '_tags'):
        tag_count = len(tnt_tagger._tags)
        print(f"  - Tag set size: {tag_count} tags")
        print(f"  - Tags: {sorted(list(tnt_tagger._tags))[:10]}...")
    
    # Theoretical complexity
    print(f"  - Theoretical complexity: O(n) per sentence")
    print(f"  - But involves: hash lookups, probability calculations, smoothing")
    
    print("\nNLTK Averaged Perceptron characteristics:")
    print(f"  - Pre-trained on large corpus")
    print(f"  - Optimized C implementations")
    print(f"  - Compiled data structures")
    print(f"  - Linear classification (very fast)")
    
    # =============================================
    # 5. SPEED RATIO ANALYSIS
    # =============================================
    print("\n5. SPEED RATIO ANALYSIS:")
    print("-" * 50)
    
    speedup = tnt_avg / nltk_avg
    tokens_per_sec_tnt = len(words) / tnt_avg
    tokens_per_sec_nltk = len(words) / nltk_avg
    
    print(f"Speed comparison:")
    print(f"  - TnT:  {tokens_per_sec_tnt:,.0f} tokens/second")
    print(f"  - NLTK: {tokens_per_sec_nltk:,.0f} tokens/second")
    print(f"  - NLTK is {1/speedup:.1f}x faster than TnT")
    
    # =============================================
    # 6. WHY IS TnT SLOWER?
    # =============================================
    print("\n6. ANALYSIS: WHY IS TnT SLOWER?")
    print("-" * 50)
    
    print("Reasons for TnT's slower performance:")
    print()
    print("1. ALGORITHM DIFFERENCES:")
    print("   • TnT: Hidden Markov Model with trigram transitions")
    print("   • NLTK: Averaged Perceptron (linear classifier)")
    print()
    print("2. IMPLEMENTATION:")
    print("   • TnT: Pure Python implementation in NLTK")
    print("   • NLTK: Cython/C optimized code")
    print()
    print("3. DATA STRUCTURES:")
    print("   • TnT: Hash table lookups for probabilities")
    print("   • NLTK: Pre-computed feature weights")
    print()
    print("4. COMPUTATION:")
    print("   • TnT: Dynamic programming, probability calculations")
    print("   • NLTK: Simple dot products")
    print()
    print("5. TRAINING DATA:")
    print("   • TnT: Trained on small Brown corpus subset")
    print("   • NLTK: Pre-trained on large WSJ corpus")
    
    return {
        'tnt_avg_time': tnt_avg,
        'nltk_avg_time': nltk_avg,
        'speedup_factor': speedup,
        'tokens_per_sec_tnt': tokens_per_sec_tnt,
        'tokens_per_sec_nltk': tokens_per_sec_nltk
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_tagger()
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_with_nltk_default()
    elif len(sys.argv) > 1 and sys.argv[1] == 'small':
        # Quick test with limited data
        train_and_benchmark_tagger('brown', max_sentences=1000)
    elif len(sys.argv) > 1 and sys.argv[1] == 'analysis':
        detailed_performance_analysis()
    else:
        # Full benchmark on large dataset
        print("Running full benchmark (this may take a few minutes)...")
        print("Use 'python pos.py small' for a quick test")
        print("Use 'python pos.py demo' for basic demo")
        print("Use 'python pos.py compare' to compare with NLTK default")
        print("Use 'python pos.py analysis' for detailed performance analysis")
        print()
        
        # Run benchmarks on different datasets
        # Start with Brown (most reliable), then try others
        datasets_to_try = ['brown']
        
        # Test if treebank is available
        try:
            from nltk.corpus import treebank
            list(treebank.tagged_sents()[:1])  # Test if we can load at least one sentence
            datasets_to_try.extend(['treebank', 'combined'])
        except Exception:
            print("Treebank not available, skipping treebank and combined tests")
        
        for dataset in datasets_to_try:
            try:
                results, tagger = train_and_benchmark_tagger(dataset)
                print(f"\n{dataset.upper()} dataset complete!")
            except Exception as e:
                print(f"Error with {dataset} dataset: {e}")
        
        # Run comparison
        try:
            compare_with_nltk_default()
        except Exception as e:
            print(f"Error in comparison: {e}") 