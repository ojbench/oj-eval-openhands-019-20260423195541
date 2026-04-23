#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move Q to SRAM once for this round
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Allocate result matrix for summing all attention outputs
    Matrix* sum_result = matrix_memory_allocator.Allocate("sum_result");
    bool first_iteration = true;
    
    // For each key-value pair up to index i
    for (size_t j = 0; j <= i; ++j) {
      // Move K[j] to SRAM for computation
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      
      // Transpose K[j] to get K[j]^T
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      
      // Compute QK^T: Q is [i+1, d], K[j]^T is [d, 1], result is [i+1, 1]
      Matrix* qk = matrix_memory_allocator.Allocate("qk");
      gpu_sim.MatMul(current_query, keys[j], qk);
      
      // Transpose K[j] back to original form
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      
      // Move K[j] back to HBM to save SRAM
      gpu_sim.MoveMatrixToGpuHbm(keys[j]);
      
      // Apply Softmax: exp(qk) / sum(exp(qk))
      Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
      gpu_sim.MatExp(qk, exp_qk);
      
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(exp_qk, sum_exp);
      
      Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result");
      gpu_sim.MatDiv(exp_qk, sum_exp, softmax_result);
      
      // Release intermediate matrices
      gpu_sim.ReleaseMatrix(qk);
      gpu_sim.ReleaseMatrix(exp_qk);
      gpu_sim.ReleaseMatrix(sum_exp);
      
      // Move V[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      
      // Compute attention output: softmax_result * V[j]
      // softmax_result is [i+1, 1], V[j] is [1, d], result is [i+1, d]
      Matrix* attention_output = matrix_memory_allocator.Allocate("attention_output");
      gpu_sim.MatMul(softmax_result, values[j], attention_output);
      
      // Release softmax_result
      gpu_sim.ReleaseMatrix(softmax_result);
      
      // Move V[j] back to HBM
      gpu_sim.MoveMatrixToGpuHbm(values[j]);
      
      // Sum with previous results
      if (first_iteration) {
        gpu_sim.Copy(attention_output, sum_result, kInSharedMemory);
        first_iteration = false;
      } else {
        Matrix* temp_sum = matrix_memory_allocator.Allocate("temp_sum");
        gpu_sim.MatAdd(sum_result, attention_output, temp_sum);
        gpu_sim.Copy(temp_sum, sum_result, kInSharedMemory);
        gpu_sim.ReleaseMatrix(temp_sum);
      }
      
      // Release attention_output
      gpu_sim.ReleaseMatrix(attention_output);
    }
    
    // Move Q back to HBM
    gpu_sim.MoveMatrixToGpuHbm(current_query);
    
    // Move final result to HBM
    gpu_sim.MoveMatrixToGpuHbm(sum_result);
    
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*sum_result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu