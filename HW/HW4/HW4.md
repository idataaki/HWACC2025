# Hardware Accelerator Assignment 4: Quantization and Pruning of CNNs

## Faculty of Engineering and Computer Science - Shahid Beheshti University

### Academic Year: 2024-2025 | Spring Semester
**Release Date:** 1404/02/13 - **Due Date:** 1404/02/23

**Instructor:** Dara Rahmati
**Assignment Designers:** Zahra Taki, Matin Firoozbakht

**Submission Format:** A `zip` file containing all required components.



## Overview

This assignment focuses on **quantization** and **pruning** techniques for Convolutional Neural Networks (CNNs). Students will implement quantization methods to reduce model size and computational complexity while maintaining acceptable accuracy levels.



## What is Quantization and Why Use It?

Quantization is a critical optimization technique in deep learning that reduces computational complexity and memory consumption. Modern deep neural networks, especially advanced models like transformers, contain billions of parameters.

### Example: GPT-3
- **Parameters:** 175 billion
- **Storage (32-bit):** ~700 GB
- **Calculation:** `(175 × 10^9 × 32) / (8 × 10^9) = 700 GB`

This massive storage requirement makes deployment on personal computers or mobile devices nearly impossible without optimization.

### Computational Efficiency

As shown in Figure 1 (from reference [1]), integer operations are significantly faster and more energy-efficient than floating-point operations on digital hardware. Quantization converts Results in reducing model size, faster inference and lower memory usage.

![area and energy efficiency of different precision](/HWACC2025/pics/area-energy-different-precision.png)*Figure 1*



## Quantization Methods

### 1. Symmetric Quantization

- Assumes data is distributed symmetrically around zero.
- For 8-bit signed: [-127, +127]
- Only requires *scale parameter*
- *Zero-point* is always 0

**Formula:**
```
q = round(x / scale)
x = q × scale

scale = max(|x|) / 127
```

**Example:**

Given tensor: `[43.21, -44.93, 0, 22.99, -43.93, -11.35, 38.48, -20.49, -38.61, -28.02]`

- Max absolute value: 44.93
- Scale: `44.93 / 127 ≈ 0.3538`
- Quantized: `[122, -127, 0, 65, -124, -32, 109, -58, -109, -79]`

**Use Case:** Best for weights with symmetric distribution around zero (e.g., neural network weights).



### 2. Asymmetric Quantization

- For data with non-symmetric distribution and specific bias.
- For 8-bit unsigned: [0, 255]
- Requires two parameters: **scale** and **zero-point**
- Zero-point maps floating-point zero to an integer value

**Formula:**
```
q = round((x - x_min) / scale)
x = q × scale + x_min

scale = (x_max - x_min) / 255
```

**Example:**

Same tensor as above:
- x_min = -44.93, x_max = 43.21
- Scale: `88.14 / 255 ≈ 0.3457`
- Zero-point: 130
- Quantized: `[255, 0, 130, 196, 3, 97, 241, 71, 18, 49]`

**Use Case:** Better for activations and layer outputs with biased or non-symmetric distributions.



## Post-Training Quantization (PTQ)

Quantization is applied **after** the model is fully trained with high precision.

**Process:**
1. Train model with 32-bit floating-point parameters
2. After training completes, convert parameters to lower precision (e.g., 8-bit integers)
3. Deploy the quantized model for inference
4. No need to retrain the model

**Goal:** Minimize model size and improve inference speed while maintaining acceptable accuracy.



## Implementation

### Step 1: Manual Implementation (Without Libraries)

**Objective:** Implement symmetric and asymmetric quantization from scratch.

**Requirements:**
- Implement both symmetric and asymmetric quantization **manually**
- No neural network libraries allowed (TensorFlow, PyTorch, etc.)
- NumPy is allowed for basic operations
- Use provided data samples

**Quantization Levels to Test:**
1. 16-bit integers
2. 8-bit integers
3. 4-bit integers

**Deliverables:**
- Working quantization functions
- Accuracy evaluation at each precision level
- Plot: Accuracy vs. Bit-width
  - X-axis: Precision level (bit-width)
  - Y-axis: Model accuracy



### Step 2: PyTorch-Based Quantization

**Objective:** Use PyTorch quantization tools to apply post-training quantization.

**Requirements:**
- Train LeNet-5 on Fashion-MNIST using PyTorch
- Apply quantization at different precision levels
- Extract weights for hardware implementation

**Precision Levels to Test:**
1. **32-bit floating-point** (baseline)
2. **16-bit floating-point**
3. **16-bit integer**
4. **8-bit integer**
5. **4-bit integer**
6. **2-bit integer**

**Deliverables:**
- Trained and quantized models at all precision levels
- Extracted weights (for hardware implementation)
- Accuracy comparison table
- Plot: Accuracy vs. Bit-width (similar to Step 1)



## Evaluation Criteria

### Step 1: Manual Quantization
- Correct implementation of symmetric quantization
- Correct implementation of asymmetric quantization
- Testing on 16-bit, 8-bit, and 4-bit precision
- Accuracy evaluation and plotting
- Code quality and documentation

### Step 2: PyTorch Quantization
- Proper model training and quantization
- Testing across all precision levels (32-bit float down to 2-bit int)
- Weight extraction for hardware use
- Accuracy comparison across all levels
- Analysis of accuracy degradation vs. compression ratio

### Analysis Questions
- How does quantization affect model accuracy?
- What is the trade-off between compression and accuracy?
- Which quantization method (symmetric vs. asymmetric) works better for this task?
- At what bit-width does accuracy significantly degrade?



## Key Learning Outcomes

- Understanding quantization fundamentals
- Implementing symmetric and asymmetric quantization
- Analyzing accuracy vs. compression trade-offs
- Working with different numerical precisions
- Using PyTorch quantization tools
- Preparing models for hardware deployment

---

## References

[1] M. Horowitz, "Computing's energy problem (and what we can do about it)", *Proc. IEEE Int. Solid-State Circuits Conf. Dig. Tech. Papers*, pp. 10-14, 2014. doi: 10.1109/ISSCC.2014.6757323

[2] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, "A Survey of Quantization Methods for Efficient Neural Network Inference," *CoRR abs/2103* (2021): 13630. doi: 10.1201/9781003162810-13

[3] Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist


