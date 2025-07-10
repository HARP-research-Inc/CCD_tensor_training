#include "tnt_tagger.hpp"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cctype>

namespace fastparse {

TnTTagger::TnTTagger() {
    // Initialize with common special tags
    tags_.insert("<S>");  // Start of sentence
    tags_.insert("</S>"); // End of sentence
    tags_.insert("UNK");  // Unknown word
}

void TnTTagger::train(const std::vector<TaggedSentence>& training_data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear existing data
    word_tag_counts_.clear();
    tag_counts_.clear();
    bigram_counts_.clear();
    trigram_counts_.clear();
    
    std::cout << "Training TnT tagger on " << training_data.size() << " sentences..." << std::endl;
    
    // Pass 1: Count unigrams, bigrams, and word-tag pairs
    for (const auto& sentence : training_data) {
        if (sentence.empty()) continue;
        
        // Add sentence boundaries
        std::vector<std::string> sentence_tags = {"<S>", "<S>"};
        
        for (const auto& word_tag : sentence) {
            const std::string& word = word_tag.first;
            const std::string& tag = word_tag.second;
            
            // Collect tags and word-tag counts
            tags_.insert(tag);
            word_tag_counts_[to_lower(word)][tag]++;
            tag_counts_[tag]++;
            sentence_tags.push_back(tag);
        }
        
        sentence_tags.push_back("</S>");
        
        // Count bigrams and trigrams
        for (size_t i = 0; i < sentence_tags.size() - 1; ++i) {
            if (i > 0) {
                bigram_counts_[sentence_tags[i-1]][sentence_tags[i]]++;
            }
            if (i > 1) {
                std::string trigram_key = make_trigram_key(
                    sentence_tags[i-2], sentence_tags[i-1], sentence_tags[i]);
                trigram_counts_[make_bigram_key(sentence_tags[i-2], sentence_tags[i-1])][sentence_tags[i]]++;
            }
        }
    }
    
    // Compute all probabilities
    compute_probabilities();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    training_time_ = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "Training completed in " << training_time_ << " seconds" << std::endl;
    std::cout << "Vocabulary size: " << vocabulary_size() << " words" << std::endl;
    std::cout << "Tag set size: " << tag_count() << " tags" << std::endl;
}

std::vector<std::pair<std::string, std::string>> TnTTagger::tag(const Sentence& words) const {
    if (words.empty()) {
        return {};
    }
    
    // Use Viterbi algorithm for optimal tagging
    std::vector<std::string> predicted_tags = viterbi_decode(words);
    
    // Combine words with predicted tags
    std::vector<std::pair<std::string, std::string>> result;
    result.reserve(words.size());
    
    for (size_t i = 0; i < words.size(); ++i) {
        result.emplace_back(words[i], predicted_tags[i]);
    }
    
    return result;
}

void TnTTagger::compute_probabilities() {
    compute_emission_probabilities();
    compute_transition_probabilities();
    compute_suffix_probabilities();
}

void TnTTagger::compute_emission_probabilities() {
    emission_probs_.clear();
    
    for (const auto& word_entry : word_tag_counts_) {
        const std::string& word = word_entry.first;
        const auto& tag_counts = word_entry.second;
        
        int total_word_count = 0;
        for (const auto& tag_count : tag_counts) {
            total_word_count += tag_count.second;
        }
        
        for (const auto& tag_count : tag_counts) {
            const std::string& tag = tag_count.first;
            int count = tag_count.second;
            
            // Simple relative frequency with add-one smoothing
            double prob = static_cast<double>(count + 1) / (total_word_count + tags_.size());
            emission_probs_[word][tag] = prob;
        }
    }
}

void TnTTagger::compute_transition_probabilities() {
    transition_probs_.clear();
    
    // Compute trigram probabilities with linear interpolation smoothing
    for (const auto& bigram_entry : trigram_counts_) {
        const std::string& bigram_key = bigram_entry.first;
        const auto& next_tag_counts = bigram_entry.second;
        
        int total_trigram_count = 0;
        for (const auto& tag_count : next_tag_counts) {
            total_trigram_count += tag_count.second;
        }
        
        for (const auto& tag_count : next_tag_counts) {
            const std::string& tag3 = tag_count.first;
            int trigram_count = tag_count.second;
            
            // Extract tag1 and tag2 from bigram_key
            size_t delimiter_pos = bigram_key.find('|');
            std::string tag1 = bigram_key.substr(0, delimiter_pos);
            std::string tag2 = bigram_key.substr(delimiter_pos + 1);
            
            // Linear interpolation smoothing
            double trigram_prob = static_cast<double>(trigram_count) / total_trigram_count;
            
            double bigram_prob = 0.0;
            if (bigram_counts_.count(tag2) && bigram_counts_.at(tag2).count(tag3)) {
                int bigram_count = bigram_counts_.at(tag2).at(tag3);
                int tag2_count = tag_counts_.count(tag2) ? tag_counts_.at(tag2) : 1;
                bigram_prob = static_cast<double>(bigram_count) / tag2_count;
            }
            
            double unigram_prob = 0.0;
            if (tag_counts_.count(tag3)) {
                int total_tags = 0;
                for (const auto& tag_count_pair : tag_counts_) {
                    total_tags += tag_count_pair.second;
                }
                unigram_prob = static_cast<double>(tag_counts_.at(tag3)) / total_tags;
            }
            
            // Linear interpolation
            double final_prob = LAMBDA1 * trigram_prob + LAMBDA2 * bigram_prob + LAMBDA3 * unigram_prob;
            transition_probs_[bigram_key][tag3] = final_prob;
        }
    }
}

void TnTTagger::compute_suffix_probabilities() {
    suffix_probs_.clear();
    
    // Compute suffix probabilities for unknown word handling
    std::unordered_map<std::string, std::unordered_map<std::string, int>> suffix_tag_counts;
    
    for (const auto& word_entry : word_tag_counts_) {
        const std::string& word = word_entry.first;
        const auto& tag_counts = word_entry.second;
        
        for (int len = 1; len <= std::min(MAX_SUFFIX_LENGTH, static_cast<int>(word.length())); ++len) {
            std::string suffix = get_suffix(word, len);
            
            for (const auto& tag_count : tag_counts) {
                const std::string& tag = tag_count.first;
                int count = tag_count.second;
                suffix_tag_counts[suffix][tag] += count;
            }
        }
    }
    
    // Convert counts to probabilities
    for (const auto& suffix_entry : suffix_tag_counts) {
        const std::string& suffix = suffix_entry.first;
        const auto& tag_counts = suffix_entry.second;
        
        int total_suffix_count = 0;
        for (const auto& tag_count : tag_counts) {
            total_suffix_count += tag_count.second;
        }
        
        if (total_suffix_count >= MIN_SUFFIX_COUNT) {
            for (const auto& tag_count : tag_counts) {
                const std::string& tag = tag_count.first;
                int count = tag_count.second;
                
                double prob = static_cast<double>(count) / total_suffix_count;
                suffix_probs_[suffix][tag] = prob;
            }
        }
    }
}

std::vector<std::string> TnTTagger::viterbi_decode(const Sentence& words) const {
    if (words.empty()) return {};
    
    const int n = words.size();
    std::vector<std::string> tags_list(tags_.begin(), tags_.end());
    const int num_tags = tags_list.size();
    
    // Viterbi matrices
    std::vector<std::vector<double>> viterbi(n, std::vector<double>(num_tags, -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> backtrack(n, std::vector<int>(num_tags, -1));
    
    // Initialize first position
    for (int t = 0; t < num_tags; ++t) {
        const std::string& tag = tags_list[t];
        double emission_prob = get_emission_prob(words[0], tag);
        double transition_prob = get_transition_prob("<S>", "<S>", tag);
        
        if (emission_prob > 0 && transition_prob > 0) {
            viterbi[0][t] = std::log(emission_prob) + std::log(transition_prob);
        }
    }
    
    // Forward pass
    for (int i = 1; i < n; ++i) {
        for (int curr_tag = 0; curr_tag < num_tags; ++curr_tag) {
            const std::string& current_tag = tags_list[curr_tag];
            double emission_prob = get_emission_prob(words[i], current_tag);
            
            if (emission_prob <= 0) continue;
            double log_emission = std::log(emission_prob);
            
            for (int prev_tag = 0; prev_tag < num_tags; ++prev_tag) {
                if (viterbi[i-1][prev_tag] == -std::numeric_limits<double>::infinity()) continue;
                
                const std::string& previous_tag = tags_list[prev_tag];
                std::string prev_prev_tag = (i > 1) ? tags_list[backtrack[i-1][prev_tag]] : "<S>";
                
                double transition_prob = get_transition_prob(prev_prev_tag, previous_tag, current_tag);
                if (transition_prob <= 0) continue;
                
                double score = viterbi[i-1][prev_tag] + std::log(transition_prob) + log_emission;
                
                if (score > viterbi[i][curr_tag]) {
                    viterbi[i][curr_tag] = score;
                    backtrack[i][curr_tag] = prev_tag;
                }
            }
        }
    }
    
    // Find best final state
    int best_final_tag = 0;
    double best_score = viterbi[n-1][0];
    for (int t = 1; t < num_tags; ++t) {
        if (viterbi[n-1][t] > best_score) {
            best_score = viterbi[n-1][t];
            best_final_tag = t;
        }
    }
    
    // Backtrack to find best path
    std::vector<std::string> result(n);
    int current_tag = best_final_tag;
    
    for (int i = n - 1; i >= 0; --i) {
        result[i] = tags_list[current_tag];
        if (i > 0) {
            current_tag = backtrack[i][current_tag];
        }
    }
    
    return result;
}

double TnTTagger::get_emission_prob(const std::string& word, const std::string& tag) const {
    std::string lower_word = to_lower(word);
    
    // Check if word is known
    if (emission_probs_.count(lower_word) && emission_probs_.at(lower_word).count(tag)) {
        return emission_probs_.at(lower_word).at(tag);
    }
    
    // Handle unknown words
    return get_unknown_word_prob(word, tag);
}

double TnTTagger::get_transition_prob(const std::string& tag1, const std::string& tag2, const std::string& tag3) const {
    std::string bigram_key = make_bigram_key(tag1, tag2);
    
    if (transition_probs_.count(bigram_key) && transition_probs_.at(bigram_key).count(tag3)) {
        return transition_probs_.at(bigram_key).at(tag3);
    }
    
    // Fallback to uniform distribution over known tags
    return 1.0 / tags_.size();
}

double TnTTagger::get_unknown_word_prob(const std::string& word, const std::string& tag) const {
    // Try suffix-based probability
    for (int len = 1; len <= std::min(MAX_SUFFIX_LENGTH, static_cast<int>(word.length())); ++len) {
        std::string suffix = get_suffix(word, len);
        if (suffix_probs_.count(suffix) && suffix_probs_.at(suffix).count(tag)) {
            return suffix_probs_.at(suffix).at(tag);
        }
    }
    
    // Heuristic-based fallback
    if (is_capitalized(word)) {
        if (tag == "NNP" || tag == "NNPS") return 0.7;
        if (tag == "NN" || tag == "NNS") return 0.2;
    }
    
    if (contains_digit(word)) {
        if (tag == "CD") return 0.8;
        if (tag == "NN") return 0.1;
    }
    
    if (contains_hyphen(word)) {
        if (tag == "JJ") return 0.5;
        if (tag == "NN") return 0.3;
    }
    
    // Default uniform probability
    return 1.0 / tags_.size();
}

// Utility methods
std::string TnTTagger::to_lower(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string TnTTagger::make_trigram_key(const std::string& tag1, const std::string& tag2, const std::string& tag3) const {
    return tag1 + "|" + tag2 + "|" + tag3;
}

std::string TnTTagger::make_bigram_key(const std::string& tag1, const std::string& tag2) const {
    return tag1 + "|" + tag2;
}

std::string TnTTagger::get_suffix(const std::string& word, int length) const {
    if (length >= static_cast<int>(word.length())) {
        return word;
    }
    return word.substr(word.length() - length);
}

bool TnTTagger::is_capitalized(const std::string& word) const {
    return !word.empty() && std::isupper(word[0]);
}

bool TnTTagger::contains_digit(const std::string& word) const {
    return std::any_of(word.begin(), word.end(), ::isdigit);
}

bool TnTTagger::contains_hyphen(const std::string& word) const {
    return word.find('-') != std::string::npos;
}

void TnTTagger::save_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw TnTException("Cannot open file for writing: " + filename);
    }
    
    // This is a simplified version - in production you'd use a proper serialization format
    file << "TnT_Model_v1.0\n";
    file << tags_.size() << "\n";
    for (const auto& tag : tags_) {
        file << tag << "\n";
    }
    
    // Save emission probabilities
    file << emission_probs_.size() << "\n";
    for (const auto& word_entry : emission_probs_) {
        file << word_entry.first << " " << word_entry.second.size() << "\n";
        for (const auto& tag_prob : word_entry.second) {
            file << tag_prob.first << " " << tag_prob.second << "\n";
        }
    }
    
    file.close();
}

void TnTTagger::load_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw TnTException("Cannot open file for reading: " + filename);
    }
    
    std::string header;
    std::getline(file, header);
    if (header != "TnT_Model_v1.0") {
        throw TnTException("Invalid model file format");
    }
    
    // Load tags
    size_t num_tags;
    file >> num_tags;
    tags_.clear();
    for (size_t i = 0; i < num_tags; ++i) {
        std::string tag;
        file >> tag;
        tags_.insert(tag);
    }
    
    // Load emission probabilities (simplified)
    size_t num_words;
    file >> num_words;
    emission_probs_.clear();
    for (size_t i = 0; i < num_words; ++i) {
        std::string word;
        size_t num_tag_probs;
        file >> word >> num_tag_probs;
        
        for (size_t j = 0; j < num_tag_probs; ++j) {
            std::string tag;
            double prob;
            file >> tag >> prob;
            emission_probs_[word][tag] = prob;
        }
    }
    
    file.close();
}

} // namespace fastparse 