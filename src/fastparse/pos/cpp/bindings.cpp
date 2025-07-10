#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "tnt_tagger.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fastpos_cpp, m) {
    m.doc() = "High-performance C++ POS tagger with Python bindings";
    
    // Main TnT Tagger class
    py::class_<fastparse::TnTTagger>(m, "TnTTagger")
        .def(py::init<>(), "Initialize a new TnT tagger")
        
        // Main methods
        .def("train", &fastparse::TnTTagger::train,
             "Train the tagger on a list of tagged sentences",
             py::arg("training_data"))
        
        .def("tag", &fastparse::TnTTagger::tag,
             "Tag a sentence (list of words) and return word-tag pairs",
             py::arg("words"))
        
        // Statistics and performance
        .def("vocabulary_size", &fastparse::TnTTagger::vocabulary_size,
             "Get the size of the vocabulary")
        
        .def("tag_count", &fastparse::TnTTagger::tag_count,
             "Get the number of unique POS tags")
        
        .def("training_time", &fastparse::TnTTagger::training_time,
             "Get the training time in seconds")
        
        // Model persistence
        .def("save_model", &fastparse::TnTTagger::save_model,
             "Save the trained model to a file",
             py::arg("filename"))
        
        .def("load_model", &fastparse::TnTTagger::load_model,
             "Load a trained model from a file",
             py::arg("filename"))
        
        // String representation
        .def("__repr__", [](const fastparse::TnTTagger& tagger) {
            return "<TnTTagger: " + std::to_string(tagger.vocabulary_size()) + 
                   " words, " + std::to_string(tagger.tag_count()) + " tags>";
        });
    
    // Exception handling
    py::register_exception<fastparse::TnTException>(m, "TnTException");
    
    // Utility functions for data conversion
    m.def("convert_nltk_corpus", [](const py::list& nltk_corpus) {
        std::vector<fastparse::TnTTagger::TaggedSentence> result;
        
        for (auto sentence : nltk_corpus) {
            fastparse::TnTTagger::TaggedSentence tagged_sentence;
            
            for (auto word_tag_pair : sentence) {
                auto pair = word_tag_pair.cast<py::tuple>();
                std::string word = pair[0].cast<std::string>();
                std::string tag = pair[1].cast<std::string>();
                tagged_sentence.emplace_back(word, tag);
            }
            
            result.push_back(tagged_sentence);
        }
        
        return result;
    }, "Convert NLTK corpus format to C++ format", py::arg("nltk_corpus"));
    
    // Benchmark utility
    m.def("benchmark_tagging", [](fastparse::TnTTagger& tagger, 
                                  const std::vector<std::vector<std::string>>& test_sentences,
                                  int iterations = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            for (const auto& sentence : test_sentences) {
                tagger.tag(sentence);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        int total_tokens = 0;
        for (const auto& sentence : test_sentences) {
            total_tokens += sentence.size();
        }
        total_tokens *= iterations;
        
        return py::dict("total_time"_a=elapsed,
                       "tokens_per_second"_a=total_tokens / elapsed,
                       "sentences_per_second"_a=(test_sentences.size() * iterations) / elapsed,
                       "avg_time_per_sentence"_a=elapsed / (test_sentences.size() * iterations));
    }, "Benchmark tagging performance", 
       py::arg("tagger"), py::arg("test_sentences"), py::arg("iterations") = 100);
} 