// Runtime Benchmark - Phase 8.1
// Tests actual concurrency behavior and performance

#include "vie/runtime/runtime.hpp"
#include "vie/model/model_config.hpp"
#include "vie/core/tensor.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>

namespace {

struct BenchmarkResult {
  std::string scenario;
  std::string backend_type;
  int worker_count;
  int request_count;
  int warmup_count;
  double total_time_ms;
  double throughput_rps;
  double avg_latency_ms;
  double p50_latency_ms;
  double p95_latency_ms;
};

void print_result(const BenchmarkResult& r) {
  std::cout << "\n=== " << r.scenario << " ===\n";
  std::cout << "Backend: " << r.backend_type << "\n";
  std::cout << "Workers: " << r.worker_count << "\n";
  std::cout << "Requests: " << r.request_count << "\n";
  std::cout << "Warmup: " << r.warmup_count << "\n";
  std::cout << "Total time: " << std::fixed << std::setprecision(2) << r.total_time_ms << " ms\n";
  std::cout << "Throughput: " << std::fixed << std::setprecision(2) << r.throughput_rps << " req/s\n";
  std::cout << "Avg latency: " << std::fixed << std::setprecision(3) << r.avg_latency_ms << " ms\n";
  std::cout << "P50 latency: " << std::fixed << std::setprecision(3) << r.p50_latency_ms << " ms\n";
  std::cout << "P95 latency: " << std::fixed << std::setprecision(3) << r.p95_latency_ms << " ms\n";
}

double percentile(std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::sort(data.begin(), data.end());
  double index = p / 100.0 * (data.size() - 1);
  size_t lower = static_cast<size_t>(index);
  size_t upper = lower + 1;
  if (upper >= data.size()) upper = data.size() - 1;
  double weight = index - lower;
  return data[lower] * (1.0 - weight) + data[upper] * weight;
}

}  // namespace

int main() {
  std::cout << "VIE Runtime Benchmark - Phase 8.1\n";
  std::cout << "==================================\n";

  // Note: Real model benchmarks require actual ONNX models
  // This benchmark demonstrates the testing methodology
  
  std::cout << "\n[Note] Real model benchmarks require ONNX model files.\n";
  std::cout << "This demonstrates the benchmark methodology with mock workloads.\n";

  // Scenario A: Single-threaded sequential (baseline)
  {
    std::cout << "\n\nScenario A: Single-threaded sequential workload\n";
    
    std::vector<double> latencies;
    int request_count = 1000;
    
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < request_count; ++i) {
      auto req_start = std::chrono::steady_clock::now();
      // Simulate work
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      auto req_end = std::chrono::steady_clock::now();
      latencies.push_back(std::chrono::duration<double, std::milli>(req_end - req_start).count());
    }
    auto end = std::chrono::steady_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    BenchmarkResult result;
    result.scenario = "A: Sequential";
    result.backend_type = "Mock";
    result.worker_count = 1;
    result.request_count = request_count;
    result.warmup_count = 0;
    result.total_time_ms = total_ms;
    result.throughput_rps = request_count / (total_ms / 1000.0);
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.p50_latency_ms = percentile(latencies, 50);
    result.p95_latency_ms = percentile(latencies, 95);
    
    print_result(result);
  }

  // Scenario B: Different models concurrent
  {
    std::cout << "\n\nScenario B: Different models concurrent (simulated)\n";
    
    vie::RuntimeConfig config;
    config.thread_pool_size = 4;
    vie::Runtime runtime(config);
    
    std::vector<double> latencies;
    int request_count = 100;
    std::mutex lat_mutex;
    
    auto start = std::chrono::steady_clock::now();
    
    std::vector<std::future<void>> futures;
    for (int i = 0; i < request_count; ++i) {
      futures.push_back(runtime.thread_pool().submit([i, &latencies, &lat_mutex] {
        auto req_start = std::chrono::steady_clock::now();
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto req_end = std::chrono::steady_clock::now();
        
        double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
        std::lock_guard<std::mutex> lock(lat_mutex);
        latencies.push_back(latency);
      }));
    }
    
    for (auto& f : futures) {
      f.get();
    }
    
    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    BenchmarkResult result;
    result.scenario = "B: Concurrent (4 workers)";
    result.backend_type = "Mock";
    result.worker_count = 4;
    result.request_count = request_count;
    result.warmup_count = 0;
    result.total_time_ms = total_ms;
    result.throughput_rps = request_count / (total_ms / 1000.0);
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.p50_latency_ms = percentile(latencies, 50);
    result.p95_latency_ms = percentile(latencies, 95);
    
    print_result(result);
  }

  // Scenario C: Same model concurrent (serialized by mutex)
  {
    std::cout << "\n\nScenario C: Same model instance concurrent (serialized)\n";
    
    std::mutex model_mutex;
    std::vector<double> latencies;
    int request_count = 100;
    std::mutex lat_mutex;
    
    vie::ThreadPool pool(4);
    
    auto start = std::chrono::steady_clock::now();
    
    std::vector<std::future<void>> futures;
    for (int i = 0; i < request_count; ++i) {
      futures.push_back(pool.submit([&model_mutex, &latencies, &lat_mutex] {
        std::lock_guard<std::mutex> lock(model_mutex);
        auto req_start = std::chrono::steady_clock::now();
        // Simulate work (serialized)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto req_end = std::chrono::steady_clock::now();
        
        double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
        std::lock_guard<std::mutex> lat_lock(lat_mutex);
        latencies.push_back(latency);
      }));
    }
    
    for (auto& f : futures) {
      f.get();
    }
    
    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    BenchmarkResult result;
    result.scenario = "C: Same model (serialized)";
    result.backend_type = "Mock";
    result.worker_count = 4;
    result.request_count = request_count;
    result.warmup_count = 0;
    result.total_time_ms = total_ms;
    result.throughput_rps = request_count / (total_ms / 1000.0);
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.p50_latency_ms = percentile(latencies, 50);
    result.p95_latency_ms = percentile(latencies, 95);
    
    print_result(result);
  }

  // Scenario D: Multiple instances of same model
  {
    std::cout << "\n\nScenario D: Multiple model instances (parallel)\n";
    
    std::vector<std::mutex> model_mutexes(4);
    std::vector<double> latencies;
    int request_count = 100;
    std::mutex lat_mutex;
    
    vie::ThreadPool pool(4);
    
    auto start = std::chrono::steady_clock::now();
    
    std::vector<std::future<void>> futures;
    for (int i = 0; i < request_count; ++i) {
      int instance_idx = i % 4;
      futures.push_back(pool.submit([instance_idx, &model_mutexes, &latencies, &lat_mutex] {
        std::lock_guard<std::mutex> lock(model_mutexes[instance_idx]);
        auto req_start = std::chrono::steady_clock::now();
        // Simulate work (parallel across instances)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto req_end = std::chrono::steady_clock::now();
        
        double latency = std::chrono::duration<double, std::milli>(req_end - req_start).count();
        std::lock_guard<std::mutex> lat_lock(lat_mutex);
        latencies.push_back(latency);
      }));
    }
    
    for (auto& f : futures) {
      f.get();
    }
    
    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    BenchmarkResult result;
    result.scenario = "D: Multiple instances (parallel)";
    result.backend_type = "Mock";
    result.worker_count = 4;
    result.request_count = request_count;
    result.warmup_count = 0;
    result.total_time_ms = total_ms;
    result.throughput_rps = request_count / (total_ms / 1000.0);
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.p50_latency_ms = percentile(latencies, 50);
    result.p95_latency_ms = percentile(latencies, 95);
    
    print_result(result);
  }

  std::cout << "\n\nBenchmark complete.\n";
  std::cout << "\nNote: These are mock benchmarks. Real model benchmarks require:\n";
  std::cout << "1. Actual ONNX model files\n";
  std::cout << "2. Real image inputs\n";
  std::cout << "3. ORT CPU and/or CUDA backends\n";
  
  return 0;
}
