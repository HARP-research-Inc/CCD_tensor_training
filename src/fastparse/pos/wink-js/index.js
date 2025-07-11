var posTagger = require('wink-pos-tagger');

// Create an instance of the pos tagger.
var tagger = posTagger();

console.log('=== BATCH TAGGING EXAMPLES ===\n');

// Method 1: Batch processing with tagRawTokens() - for pre-split tokens
console.log('1. Using tagRawTokens() for batch processing:');
var rawTokens = ['I', 'love', 'machine', 'learning', 'and', 'natural', 'language', 'processing', '.'];
var result1 = tagger.tagRawTokens(rawTokens);
console.log('Input:', rawTokens);
console.log('Output:', result1.map(t => `${t.value}/${t.pos}`).join(' '));
console.log('');

// Method 2: Multiple sentences with tagSentence() in a loop
console.log('2. Processing multiple sentences:');
var sentences = [
    'He is trying to fish for fish in the lake.',
    'I love programming with JavaScript.',
    'The weather is beautiful today.',
    'Machine learning is fascinating.'
];

sentences.forEach((sentence, index) => {
    var result = tagger.tagSentence(sentence);
    console.log(`Sentence ${index + 1}: "${sentence}"`);
    console.log(`Tagged: ${result.map(t => `${t.value}/${t.pos}`).join(' ')}`);
    console.log('');
});

// Method 3: Batch processing multiple sentences and collecting results
console.log('3. Batch processing with results collection:');
var batchResults = sentences.map(sentence => {
    return {
        original: sentence,
        tagged: tagger.tagSentence(sentence),
        posSequence: tagger.tagSentence(sentence).map(t => t.pos).join(' ')
    };
});

batchResults.forEach((result, index) => {
    console.log(`Result ${index + 1}:`);
    console.log(`  Sentence: ${result.original}`);
    console.log(`  POS sequence: ${result.posSequence}`);
    console.log(`  Word count: ${result.tagged.filter(t => t.tag === 'word').length}`);
    console.log('');
});

// Method 4: Processing a large document (split by sentences)
console.log('4. Processing a document:');
var document = `Natural language processing is a field of artificial intelligence. 
It focuses on the interaction between computers and human language. 
Modern NLP uses machine learning techniques.`;

var documentSentences = document.split(/[.!?]+/).filter(s => s.trim().length > 0);
var documentResults = documentSentences.map(sentence => tagger.tagSentence(sentence.trim()));

console.log(`Document processed: ${documentSentences.length} sentences`);
documentResults.forEach((result, index) => {
    var nouns = result.filter(t => t.pos && t.pos.startsWith('N')).map(t => t.value);
    var verbs = result.filter(t => t.pos && t.pos.startsWith('V')).map(t => t.value);
    console.log(`  Sentence ${index + 1} - Nouns: [${nouns.join(', ')}], Verbs: [${verbs.join(', ')}]`);
});

console.log('\n=== PERFORMANCE OPTIMIZED BATCH PROCESSING ===');

// Method 5: Performance optimized batch processing
function batchTagSentences(sentences) {
    console.log(`Processing ${sentences.length} sentences...`);
    var startTime = Date.now();
    
    var results = sentences.map(sentence => tagger.tagSentence(sentence));
    
    var endTime = Date.now();
    var totalTokens = results.reduce((sum, result) => sum + result.length, 0);
    
    console.log(`Processed ${totalTokens} tokens in ${endTime - startTime}ms`);
    console.log(`Rate: ${Math.round(totalTokens / (endTime - startTime) * 1000)} tokens/second`);
    
    return results;
}

// Test with a larger batch
var largeBatch = [
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning algorithms can process natural language.',
    'JavaScript is a versatile programming language.',
    'Data science involves statistics and programming.',
    'Artificial intelligence is transforming many industries.'
];

var largeBatchResults = batchTagSentences(largeBatch);

console.log('\nSample output from large batch:');
console.log(largeBatchResults[0].map(t => `${t.value}/${t.pos}`).join(' '));