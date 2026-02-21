# Hardware Accelerator Assignment 5: CUDA Programming - Softmax Implementation

## Faculty of Engineering and Computer Science - Shahid Beheshti University

### Academic Year: 2024-2025 | Spring Semester
**Release Date:** 1404/02/24 - **Due Date:** 1404/03/02

**Instructor:** Dara Rahmati
**Assignment Designers:** Zahra Taki, Matin Firoozbakht

**Submission Format:** A `zip` file containing all required components.

---

## Overview

This assignment introduces **CUDA programming** for GPU acceleration. Students will implement the **Softmax activation function** using CUDA and optimize its performance by experimenting with different block sizes and thread configurations.

---

## Introduction to CUDA

**CUDA (Compute Unified Device Architecture)** is NVIDIA's parallel computing platform and programming model. It has become a key tool in high-performance computing.

### Why CUDA?

- **Parallel Processing:** Leverage GPU's massive parallel processing power
- **Speed:** Execute computationally intensive tasks faster than CPU
- **Applications:** Machine learning, image processing, scientific simulations
- **Language Support:** C/C++ programming languages

### Key Concepts

1. **Kernels:** Functions that run in parallel on the GPU
2. **Blocks:** Groups of threads that execute together
3. **Threads:** Individual execution units
4. **Hierarchical Execution:** Organized structure for efficient hardware utilization

### Learning Resources

Students should refer to the resources provided in the References section to learn CUDA programming.

---

## Softmax Activation Function

### What is Softmax?

The **Softmax function** is commonly used in the final layer of neural networks for classification tasks. It converts raw output values (logits) into interpretable probabilities.

**Mathematical Formula:**
```
Softmax(z_i) = e^(z_i) / Σ(e^(z_j)) for j=1 to K
```

Where:
- `z_i` = input for class i (logit)
- `K` = total number of classes
- Denominator = sum of exponentials of all class scores

### How Softmax Works

**Process:**
1. **Input:** Receives a vector of real numbers from the final network layer
2. **Exponentiation:** Each element raised to power of e (~2.718) to make all values positive
3. **Normalization:** Divide exponentials by their sum to get final output

**Result:**
All outputs are between 0 and 1. Sum of all outputs equals 1 and outputs can be interpreted as probabilities.

### Example

    Input vector: [3.2, 1.3, 0.2, 0.8]
    After Softmax: [0.532, 0.239, 0.107, 0.293]

### Applications

- **Multi-class Classification:** Primary use case
- **Image Recognition:** Classify images into multiple categories
- **Natural Language Processing (NLP):** Text classification and language tasks

### Sigmoid vs. Softmax
- Both outputs between 0 and 1
- Softmax is **Guaranteed** to sum to 1, Sigmoid is not

---

## Why Use CUDA for Softmax?

- **Computational Challenges:** Softmax involves computationally intensive operations such as *Exponential calculation* of input elements and *Normalization* by sum of all exponentials.

- **Sequential Nature:** Softmax Must first compute sum of all exponentials, then use that sum for normalization, This data dependency seems to prevent parallelization.

Despite the sequential nature, CUDA and GPU architecture enable parallelization as different stages of Softmax can be parallelized and also some independent operations can be split and executed in parallel.

---

## Assignment Task: Performance Evaluation

### Objective

Evaluate the performance of Softmax function in the **attention layer of Vision Transformer (ViT)** networks using CUDA and find the optimal configuration for block size and thread count.

### Context: Vision Transformers

- Softmax is extensively used in the attention mechanism of Vision Transformers
- Input vectors typically have **length 197** (for standard ViT)
- Fast and optimized execution on GPU is critical

### Performance Measurement

Implement and test with different block sizes (32, 64, 128, 256, etc. threads per block). For each block size configuration:

1. **Measure kernel execution time**
2. **Record results systematically**
3. **Create performance graphs**
4. **Analyze trade-offs**

### Analysis Requirements

**Graph Requirements:**
- X-axis: Block size (number of threads)
- Y-axis: Execution time (milliseconds or microseconds)
- Clear labels and legend

**Analysis Questions:**
1. Which block size provides the **lowest execution time**?
2. What is the **optimal configuration** for maximum GPU resource utilization?
3. How does performance scale with block size?
4. What is the trade-off between parallelism and overhead?

### Deliverables

1. **CUDA Implementation:**
   - Softmax kernel code
   - Host code for testing different configurations
   - Proper memory management (allocation, transfer, deallocation)

2. **Performance Data:**
   - Execution times for all block sizes
   - Raw data in table format

3. **Visualization:**
   - Performance graph (execution time vs. block size)
   - Clear comparison of all configurations

4. **Analysis:**
   - Identification of optimal configuration
   - Explanation of performance trends
   - Discussion of GPU resource utilization

5. **Report:**
   - Implementation details
   - Experimental methodology
   - Results and analysis
   - Conclusions and recommendations

---

## Implementation Guidelines

### CUDA Softmax Structure

**Typical Implementation Steps:**

1. **Allocate GPU memory** for input and output vectors
2. **Copy input data** from host to device
3. **Launch kernel** to compute exponentials (parallelizable)
4. **Launch kernel** to compute sum of exponentials (use reduction)
5. **Launch kernel** to normalize by dividing by sum (parallelizable)
6. **Copy results** back to host
7. **Free GPU memory**

### Optimization Considerations

- Use **shared memory** for intermediate results
- Implement **parallel reduction** for sum calculation
- Consider **numerical stability** (subtract max value before exp)

### Python Reference Implementation

Students should understand the Python implementation shown in the assignment PDF before implementing in CUDA:

```python
from math import import exp

def softmax(input_vector):
    # Calculate exponential of each element
    exponents = [exp(i) for i in input_vector]

    # Sum all exponentials
    sum_of_exponents = sum(exponents)

    # Normalize to get probabilities
    probabilities = [round(exp(i) / sum_of_exponents, 3)
                    for i in exponents]

    return probabilities
```

---

## Testing and Validation

### Correctness Verification

1. Compare CUDA output with Python/NumPy reference implementation
2. Verify that outputs sum to 1.0
3. Check all outputs are in range [0, 1]
4. Test with different input sizes

### Performance Testing

1. **Warm-up runs:** Execute kernel several times before measuring
2. **Multiple measurements:** Average over multiple runs for reliability
3. **Consistent conditions:** Same GPU state for all tests
4. **Timing method:** Use CUDA events or appropriate timing API

---

## Key Learning Outcomes

- Understanding CUDA programming fundamentals
- Implementing parallel algorithms on GPU
- Working with kernels, blocks, and threads
- Performance measurement and optimization
- Analyzing trade-offs in parallel computing
- Applying GPU acceleration to real neural network operations
- Understanding Vision Transformer architecture

---

## References

### CUDA Learning Resources

[1] "CUDA Toolkit Documentation 12.9." Nvidia.com, 2025, https://docs.nvidia.com/cuda/

[2] freeCodeCamp.org. "CUDA Programming Course – High-Performance Computing with GPUs." YouTube, 24 Sept. 2024, https://www.youtube.com/watch?v=86FAWCzIe_4

[3] Tom Nurkkala. "CUDA Programming." YouTube, 25 Nov. 2020, https://www.youtube.com/watch?v=xwbD6fL5qC8

### Softmax Resources

[4] P. Belagatti, "Understanding the Softmax Activation Function: A Comprehensive Guide," SingleStore, Mar. 11, 2024. https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/

---

## Additional Notes

### CUDA Programming Tips

- **Memory Management:** Always free allocated GPU memory
- **Error Checking:** Check CUDA API calls for errors
- **Thread Safety:** Ensure no race conditions in parallel code
- **Block Size Limits:** Maximum threads per block is typically 1024
- **Shared Memory:** Use for frequently accessed data

### Common Pitfalls to Avoid

- Not handling edge cases (empty vectors, very large values)
- Ignoring numerical stability (exp overflow)
- Incorrect memory transfer sizes
- Not synchronizing GPU operations when needed
- Forgetting to measure actual kernel time (excluding transfer time)

---

## Input Specification

**For Vision Transformer Attention Layer:**
- Vector length: **197 elements**
- Data type: Floating-point (float32)
- Values: Real numbers (can be positive or negative before softmax)

**Test Cases:**
- Random vectors of length 197
- Multiple iterations for statistical reliability
- Vary input value ranges to test robustness

---

Good luck!

**Instructor:** Rahmati
