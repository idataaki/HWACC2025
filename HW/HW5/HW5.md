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

**Key Focus:**
- Learning CUDA programming fundamentals
- Parallel programming on GPUs
- Performance optimization
- Block and thread configuration

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
- All outputs are between 0 and 1
- Sum of all outputs equals 1
- Outputs can be interpreted as probabilities

### Example

**Input vector:** `[3.2, 1.3, 0.2, 0.8]`

**After Softmax:** `[0.532, 0.239, 0.107, 0.293]`

All values are probabilities that sum to 1.0

---

## Properties and Advantages of Softmax

### Key Properties

1. **Output Range:** All outputs between 0 and 1
2. **Sum to One:** Total of all outputs equals 1
3. **Interpretability:** Converts raw scores to understandable probabilities

### Advantages

1. **Probability Distribution:** Creates precise probability distribution for each class
2. **Confidence Assessment:** Allows evaluation of network's confidence level
3. **Interpretability:** Probabilities are easier to understand and report than raw scores
4. **Numerical Stability:** Proper implementation (e.g., subtracting max value) provides good numerical stability

### Applications

- **Multi-class Classification:** Primary use case
- **Image Recognition:** Classify images into multiple categories
- **Natural Language Processing (NLP):** Text classification and language tasks
- **Recommendation Systems:** Probability-based recommendations

**Example:** In a fruit classification network, Softmax outputs the probability that an image is an apple, orange, or banana, with probabilities summing to 1.

---

## Comparison with Other Activation Functions

### Sigmoid vs. Softmax

- **Sigmoid:** Used in hidden layers or binary classification
  - Outputs between 0 and 1
  - No guarantee that outputs sum to 1

- **Softmax:** Used in output layer for multi-class classification
  - Outputs between 0 and 1
  - **Guaranteed** to sum to 1

### ReLU vs. Softmax

- **ReLU (Rectified Linear Unit):** Used in hidden layers
  - Solves vanishing gradient problem
  - Non-linear transformation

- **Softmax:** Used in output layer
  - Produces probability distribution
  - Multi-class classification

**Key Distinction:** Softmax's unique role is in multi-class classification scenarios where outputs must represent a complete probability distribution.

---

## Why Use CUDA for Softmax?

### Computational Challenges

Softmax involves two computationally intensive operations:

1. **Exponential calculation** of input elements
2. **Normalization** by sum of all exponentials

### Sequential Nature

Softmax has an inherently **sequential nature**:
- Must first compute sum of all exponentials
- Then use that sum for normalization
- Data dependency seems to prevent parallelization

### GPU Solution

Despite the sequential nature, **CUDA and GPU architecture** enable parallelization:

- **Thousands of threads** can run simultaneously
- **Different stages** of Softmax can be parallelized
- **Independent operations** can be split and executed in parallel

**Key Insight:** By breaking down Softmax into parallelizable sub-tasks, CUDA can dramatically accelerate computation.

---

## Assignment Task: Performance Evaluation

### Objective

Evaluate the performance of Softmax function in the **attention layer of Vision Transformer (ViT)** networks using CUDA and find the optimal configuration for block size and thread count.

### Context: Vision Transformers

- Softmax is extensively used in the attention mechanism of Vision Transformers
- Input vectors typically have **length 197** (for standard ViT)
- Fast and optimized execution on GPU is critical

### Implementation Requirements

**Task:** Implement Softmax in CUDA with different block configurations

**Block Sizes to Test:**
- 32 threads per block
- 64 threads per block
- 128 threads per block
- 256 threads per block
- Additional sizes as needed (512, 1024, etc.)

### Performance Measurement

For each block size configuration:

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
- Minimize **memory transfers** between host and device

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

## Vision Transformer Context

### Attention Layer in ViT

- Vision Transformers use self-attention mechanisms
- Softmax is applied to attention scores
- Input sequence length for standard ViT: **197 tokens**
  - 1 class token + 196 patch tokens (14×14 patches)

### Why Vector Length = 197?

- **Class token:** 1
- **Image patches:** 196 (from 224×224 image divided into 16×16 patches)
- **Total:** 197 elements in attention computation

This specific vector length should be used for testing to match real Vision Transformer workloads.

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

## Submission Guidelines

**File Format:** ZIP file named `HW5_StudentNumber_Fullname.zip`

**Structure:**
- CUDA source files (.cu, .cuh)
- Host code for testing
- Performance data (CSV or text file)
- Performance graphs (images)
- Report (use provided template - **mandatory**)

**Report Contents:**
- CUDA implementation explanation
- Block size configurations tested
- Performance results table
- Performance graph
- Analysis and conclusions
- Optimal configuration recommendation

**Extensions:**
- Contact TA on Telegram before requesting extension
- Students have 4 coupon days available

**Important:**
- Reports must use the provided template
- Reports outside the template will **not be reviewed**

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
