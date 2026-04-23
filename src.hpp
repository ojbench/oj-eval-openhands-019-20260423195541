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
    Matrix* sum_result = nullptr;
    
    // For each key-value pair up to index i
    for (size_t j = 0; j <= i; ++j) {
      // Move K[j] to SRAM and copy it
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix* k_copy = matrix_memory_allocator.Allocate("k_copy");
      gpu_sim.Copy(keys[j], k_copy, kInSharedMemory);
      
      // Move K[j] back to HBM
      gpu_sim.MoveMatrixToGpuHbm(keys[j]);
      
      // Transpose the copy
      gpu_sim.Transpose(k_copy, kInSharedMemory);
      
      // Compute QK^T: Q is [i+1, d], K[j]^T is [d, 1], result is [i+1, 1]
      Matrix* qk = matrix_memory_allocator.Allocate("qk");
      gpu_sim.MatMul(current_query, k_copy, qk);
      
      // Release k_copy
      gpu_sim.ReleaseMatrix(k_copy);
      
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
      
      // Move V[j] to SRAM and copy it
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix* v_copy = matrix_memory_allocator.Allocate("v_copy");
      gpu_sim.Copy(values[j], v_copy, kInSharedMemory);
      
      // Move V[j] back to HBM
      gpu_sim.MoveMatrixToGpuHbm(values[j]);
      
      // Compute attention output: softmax_result * V[j]
      // softmax_result is [i+1, 1], V[j] is [1, d], result is [i+1, d]
      Matrix* attention_output = matrix_memory_allocator.Allocate("attention_output");
      gpu_sim.MatMul(softmax_result, v_copy, attention_output);
      
      // Release intermediate matrices
      gpu_sim.ReleaseMatrix(softmax_result);
      gpu_sim.ReleaseMatrix(v_copy);
      
      // Sum with previous results
      if (sum_result == nullptr) {
        sum_result = attention_output;
      } else {
        Matrix* new_sum = matrix_memory_allocator.Allocate("new_sum");
        gpu_sim.MatAdd(sum_result, attention_output, new_sum);
        gpu_sim.ReleaseMatrix(sum_result);
        gpu_sim.ReleaseMatrix(attention_output);
        sum_result = new_sum;
      }
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