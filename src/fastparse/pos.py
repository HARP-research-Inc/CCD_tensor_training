from nltk.tag import tnt
import nltk

# Download required NLTK data if not present
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')

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

if __name__ == "__main__":
    demo_tagger()
