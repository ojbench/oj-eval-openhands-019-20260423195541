#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move Q to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Compute attention for each key-value pair and sum
    std::vector<Matrix*> attention_outputs;
    
    for (size_t j = 0; j <= i; ++j) {
      // Move K[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      
      // Transpose K[j]
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      
      // Compute QK^T
      Matrix* qk = matrix_memory_allocator.Allocate("qk");
      gpu_sim.MatMul(current_query, keys[j], qk);
      
      // Transpose K[j] back
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      
      // Move K[j] back to HBM
      gpu_sim.MoveMatrixToGpuHbm(keys[j]);
      
      // Compute softmax
      Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk");
      gpu_sim.MatExp(qk, exp_qk);
      
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(exp_qk, sum_exp);
      
      Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax");
      gpu_sim.MatDiv(exp_qk, sum_exp, softmax_result);
      
      // Move V[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      
      // Compute attention output
      Matrix* att_out = matrix_memory_allocator.Allocate("att_out");
      gpu_sim.MatMul(softmax_result, values[j], att_out);
      
      // Move V[j] back to HBM
      gpu_sim.MoveMatrixToGpuHbm(values[j]);
      
      attention_outputs.push_back(att_out);
    }
    
    // Sum all attention outputs
    Matrix* result = attention_outputs[0];
    for (size_t j = 1; j < attention_outputs.size(); ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp");
      gpu_sim.MatAdd(result, attention_outputs[j], temp);
      result = temp;
    }
    
    // Move Q back to HBM
    gpu_sim.MoveMatrixToGpuHbm(current_query);
    
    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);
    
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu