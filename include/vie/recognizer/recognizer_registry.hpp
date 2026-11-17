#pragma once

#include "vie/recognizer/model_recognizer.hpp"

#include <memory>
#include <vector>

namespace vie {

class ModelRecognizerRegistry {
 public:
  void add(std::unique_ptr<IModelRecognizer> recognizer);
  RecognitionResult recognize(const ModelAnalysisReport& report) const;

 private:
  std::vector<std::unique_ptr<IModelRecognizer>> recognizers_;
};

ModelRecognizerRegistry default_recognizer_registry();

}  // namespace vie
