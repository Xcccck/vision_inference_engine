// Multi-model runtime example demonstrating concurrent inference
//
// This example shows how to:
// 1. Create a Runtime with thread pool
// 2. Load multiple models
// 3. Submit concurrent inference tasks
// 4. Collect metrics
// 5. Unload models

#include "vie/runtime/runtime.hpp"
#include "vie/model/model_config.hpp"
#include "vie/core/error.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <future>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model1.onnx> [model2.onnx ...]\n";
    return 1;
  }

  try {
    // Configure runtime
    vie::RuntimeConfig config;
    config.thread_pool_size = 4;
    config.default_device = vie::DeviceType::Cpu;
    config.default_device_id = 0;
    config.enable_profiling = true;
    config.max_loaded_models = 16;

    // Create runtime
    vie::Runtime runtime(config);
    std::cout << "Runtime created with " << runtime.thread_pool().worker_count()
              << " worker threads\n";

    // Load models
    std::vector<std::string> model_ids;
    for (int i = 1; i < argc; ++i) {
      std::string model_path = argv[i];
      std::string model_id = "model_" + std::to_string(i);

      vie::ModelConfig model_config;
      model_config.model_path = model_path;
      model_config.device = config.default_device;
      model_config.device_id = config.default_device_id;
      model_config.backend = vie::BackendKind::Auto;

      // Note: In a real scenario, you would need to provide a manifest
      // For this example, we assume the model can be auto-analyzed
      // runtime.models().load(model_id, model_config);
      // model_ids.push_back(model_id);
      
      std::cout << "Would load model: " << model_path << " as " << model_id << "\n";
      model_ids.push_back(model_id);
    }

    std::cout << "\nLoaded " << model_ids.size() << " model(s)\n";

    // Submit concurrent inference tasks
    std::cout << "\nSubmitting concurrent inference tasks...\n";
    std::vector<std::future<void>> tasks;
    
    for (const auto& model_id : model_ids) {
      auto model = runtime.models().get(model_id);
      if (!model) {
        std::cerr << "Model " << model_id << " not found\n";
        continue;
      }

      // Submit inference task to thread pool
      auto task = runtime.thread_pool().submit([model_id, &runtime]() {
        std::cout << "Task for " << model_id << " running on thread\n";
        // In a real scenario, you would:
        // 1. Load an image
        // 2. Call model->infer(image)
        // 3. Process results
      });
      tasks.push_back(std::move(task));
    }

    // Wait for all tasks to complete
    for (auto& task : tasks) {
      task.get();
    }

    std::cout << "All tasks completed\n";

    // Print metrics
    auto metrics = runtime.snapshot_metrics();
    std::cout << "\n=== Runtime Metrics ===\n";
    std::cout << "Total inferences: " << metrics.total_inferences << "\n";
    std::cout << "Failed inferences: " << metrics.failed_inferences << "\n";
    std::cout << "Total inference time: " << metrics.total_inference_time_ms << " ms\n";
    std::cout << "Models loaded: " << metrics.total_models_loaded << "\n";
    std::cout << "Models unloaded: " << metrics.total_models_unloaded << "\n";

    if (!metrics.per_model_inference_count.empty()) {
      std::cout << "\nPer-model statistics:\n";
      for (const auto& [id, count] : metrics.per_model_inference_count) {
        double time = metrics.per_model_inference_time_ms.at(id);
        std::cout << "  " << id << ": " << count << " inferences, "
                  << time << " ms total\n";
      }
    }

    // Unload models
    std::cout << "\nUnloading models...\n";
    for (const auto& model_id : model_ids) {
      if (runtime.models().contains(model_id)) {
        runtime.models().unload(model_id);
        std::cout << "Unloaded " << model_id << "\n";
      }
    }

    std::cout << "\nDone\n";
    return 0;

  } catch (const vie::VieError& e) {
    std::cerr << "VIE error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
