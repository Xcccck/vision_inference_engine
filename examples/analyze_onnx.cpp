#include "vie/analyzer/onnx_analyzer.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: vie_analyze_onnx <model.onnx>\n";
    return 2;
  }
  try {
    const vie::OnnxAnalyzer analyzer;
    std::cout << vie::analysis_to_text(analyzer.analyze_file(argv[1]));
    return 0;
  } catch (const vie::VieError& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
