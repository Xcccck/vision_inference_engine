#include "vie/analyzer/onnx_analyzer.hpp"
#include "vie/recognizer/manifest_generator.hpp"
#include "vie/recognizer/recognizer_registry.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: vie_recognize_onnx <model.onnx>\n";
    return 2;
  }
  try {
    const vie::OnnxAnalyzer analyzer;
    const vie::ModelAnalysisReport report = analyzer.analyze_file(argv[1]);
    const vie::RecognitionResult recognition = vie::default_recognizer_registry().recognize(report);
    const vie::ModelManifest manifest =
        vie::ManifestGenerator{}.generate(report, recognition, argv[1]);
    std::cout << vie::recognition_to_text(recognition);
    std::cout << "manifest:\n" << manifest.to_json_string(2) << '\n';
    return 0;
  } catch (const vie::VieError& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
