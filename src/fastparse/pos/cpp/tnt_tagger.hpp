#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>

namespace fastparse {

/**
 * High-performance C++ implementation of TnT (Trigrams'n'Tags) POS tagger
 * 
 * Features:
 * - Optimized hash maps for O(1) probability lookups
 * - Efficient memory layout for cache-friendly access
 * - Trigram transition probabilities with smoothing
 * - Unknown word handling with suffix analysis
 */
class TnTTagger {
public:
    using TaggedSentence = std::vector<std::pair<std::string, std::string>>;
    using Sentence = std::vector<std::string>;
    
    // Constructor
    TnTTagger();
    
    // Main interface
    void train(const std::vector<TaggedSentence>& training_data);
    std::vector<std::pair<std::string, std::string>> tag(const Sentence& words) const;
    
    // Performance and statistics
    size_t vocabulary_size() const { return word_tag_counts_.size(); }
    size_t tag_count() const { return tags_.size(); }
    double training_time() const { return training_time_; }
    
    // Serialization (for saving/loading trained models)
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
private:
    // Core data structures
    std::unordered_set<std::string> tags_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_tag_counts_;
    std::unordered_map<std::string, int> tag_counts_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> bigram_counts_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> trigram_counts_;
    
    // Probabilities (computed after training)
    std::unordered_map<std::string, std::unordered_map<std::string, double>> emission_probs_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> transition_probs_;
    
    // Unknown word handling
    std::unordered_map<std::string, std::unordered_map<std::string, double>> suffix_probs_;
    
    // Configuration
    static constexpr double LAMBDA1 = 0.8;  // Trigram weight
    static constexpr double LAMBDA2 = 0.15; // Bigram weight  
    static constexpr double LAMBDA3 = 0.05; // Unigram weight
    static constexpr int MIN_SUFFIX_COUNT = 5;
    static constexpr int MAX_SUFFIX_LENGTH = 4;
    
    // Performance tracking
    mutable double training_time_ = 0.0;
    
    // Helper methods
    void compute_probabilities();
    void compute_emission_probabilities();
    void compute_transition_probabilities();
    void compute_suffix_probabilities();
    
    double get_emission_prob(const std::string& word, const std::string& tag) const;
    double get_transition_prob(const std::string& tag1, const std::string& tag2, const std::string& tag3) const;
    double get_unknown_word_prob(const std::string& word, const std::string& tag) const;
    
    std::string get_suffix(const std::string& word, int length) const;
    bool is_capitalized(const std::string& word) const;
    bool contains_digit(const std::string& word) const;
    bool contains_hyphen(const std::string& word) const;
    
    // Viterbi decoding
    std::vector<std::string> viterbi_decode(const Sentence& words) const;
    
    // String processing utilities
    std::string to_lower(const std::string& str) const;
    std::string make_trigram_key(const std::string& tag1, const std::string& tag2, const std::string& tag3) const;
    std::string make_bigram_key(const std::string& tag1, const std::string& tag2) const;
};

// Exception class for TnT-specific errors
class TnTException : public std::exception {
public:
    explicit TnTException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
    
private:
    std::string message_;
};

} // namespace fastparse 