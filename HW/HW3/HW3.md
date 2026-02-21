# Hardware Accelerator Assignment 3: Convolutional Neural Network with Systolic Array

## Faculty of Engineering and Computer Science - Shahid Beheshti University

### Academic Year: 2024-2025 | Spring Semester
**Release Date:** 1404/01/27 - **Due Date:** 1404/02/12

**Instructor:** Dara Rahmati
**Assignment Designers:** Zahra Taki, Matin Firoozbakht

**Submission Format:** A `zip` file containing all required components.

---

## Overview

This assignment focuses on implementing a **Convolutional Neural Network (LeNet)** using **systolic array architecture** with different dataflows. Students will explore how different dataflow strategies affect accelerator performance and hardware resource utilization.

CNNs are crucial for image processing tasks including classification, detection, and segmentation. The high computational cost of these networks makes hardware acceleration particularly important.

---

## LeNet Network Architecture

The LeNet architecture is detailed in Table 2.2 of the reference textbook. The network consists of:

| Layer | Type | Channels | Size | Kernel | Stride | Activation |
|-------|------|----------|------|--------|--------|------------|
| Input | Image | 1 | 32×32 | - | - | - |
| 1 | Convolution | 6 | 28×28 | 5×5 | 1 | tanh |
| 2 | AveragePool | 6 | 14×14 | 2×2 | 2 | tanh |
| 3 | Convolution | 16 | 10×10 | 5×5 | 1 | tanh |
| 4 | AveragePool | 16 | 5×5 | 2×2 | 2 | tanh |
| 5 | Convolution | 120 | 1×1 | 5×5 | 1 | tanh |
| 6 | FC (Fully Connected) | - | 84 | - | - | tanh |
| Output | FC | - | 10 | - | - | Softmax |

**Key Features:**
- Two convolutional layers with 5×5 kernels
- Two average pooling layers
- Two fully connected layers
- Total parameters: ~60,000 weights

---

## Dataset

**Fashion MNIST Dataset**
- **Training samples:** 60,000 images
- **Test samples:** 10,000 images
- **Image size:** 28×28 grayscale
- **Classes:** 10 clothing categories
- **Source:** https://github.com/zalandoresearch/fashion-mnist

**Important Note:**
LeNet expects 32×32 input, so you must apply **zero-padding** to convert 28×28 images to 32×32. This padding must be applied in both software training and hardware implementation.

---

## Dataflows

Three fundamental dataflow strategies will be implemented (see Section 5.6 of the reference textbook):

1. **Weight Stationary (WS)** - Weights remain fixed in PEs, inputs and outputs flow
2. **Output Stationary (OS)** - Partial sums remain in PEs, weights and inputs flow
3. **Input Stationary (IS)** - Input activations remain in PEs, weights and outputs flow

Each dataflow affects the accelerator architecture differently. Reference Table 5.1 in the textbook for architectures using these dataflows.

---

## Systolic Array Architecture

The systolic array is a common architecture for general matrix multiplication and neural network acceleration. In this assignment:

- Processing Elements (PEs) are arranged in a systolic array configuration
- PEs perform MAC (Multiply-Accumulate) operations
- The array size should be **5×5** to match the kernel size
- Dataflow determines how data moves through the array

**Example:** Figure 5.16 from the textbook shows the WS dataflow in NeuFlow, where PEs compute MAC operations and are connected in a systolic array to calculate partial sums.

---

## Memory Architecture

**Two-level memory hierarchy:**

1. **Level 1 - Weight Memory (DRAM model):**
   - Stores all network weights (~60,000 parameters)
   - Low-speed memory
   - Design size based on extracted weights from training

2. **Level 2 - Activation/Output Memory:**
   - Stores partial sums and intermediate outputs
   - Separate from weight memory

Load weights into memory modules and execute the network using these stored weights.

---

## Part 1: Software Implementation

### Objective
Train the LeNet network and extract weights for hardware implementation.

### Requirements

1. **Framework:** TensorFlow or PyTorch
2. **Dataset:** Fashion MNIST with zero-padding to 32×32
3. **Training:** Train the network and achieve good accuracy
4. **Weight Extraction:** Save trained weights for hardware implementation

### Deliverables
- Trained LeNet model
- Extracted weight files
- Training accuracy report

---

## Part 2: Hardware Implementation

### Objective
Implement LeNet inference in hardware with three different dataflows using systolic array architecture.

### Three Implementations Required

You must create **three separate hardware implementations**, each using one dataflow:
1. Weight Stationary (WS) implementation
2. Output Stationary (OS) implementation
3. Input Stationary (IS) implementation

### Design Specifications

**Common Requirements:**
- Hardware Description Language: Verilog/VHDL
- Design Tool: Xilinx Vivado
- Inference only (no training in hardware)
- Use weights from software implementation
- Systolic array size: 5×5 PEs
- Each PE performs MAC operations

**Memory Implementation:**
- Implement weight memory (DRAM model - slow access)
- Implement activation/output memory
- Load trained weights into memory
- Read data from memory during inference

**Vivado Requirements:**
- Successfully complete synthesis and implementation
- Generate timing reports
- Extract power consumption metrics
- Report schematic diagrams

### Notes
- Zero-padding can be done on original images or during hardware preprocessing
- Read dataset files directly in HDL code

---

## Part 3: Evaluation

### Evaluation 1: Speed Comparison

**Task:** Compare inference time between software and hardware implementations.

Create a table comparing inference time for:
- Software implementation (TensorFlow/PyTorch)
- Hardware implementation (WS dataflow)
- Hardware implementation (OS dataflow)
- Hardware implementation (IS dataflow)

**Analysis Questions:**
1. How do you calculate hardware inference time?
2. Why is this comparison not fair?
3. What do you suggest for a fair speed comparison between accelerators?

### Evaluation 2: Hardware Comparison

**Task:** Compare the three hardware implementations.

After synthesis and implementation in Vivado, extract and compare:
- **Total on-chip power consumption**
- **Throughput** (bits processed per unit time)

**Throughput Calculation:**
- Extract timing parameters from Vivado reports
- Calculate throughput using timing settings
- Document calculation methodology

**Analysis:**
- Which dataflow provides best throughput?
- Which dataflow is most power-efficient?
- Discuss trade-offs between the three approaches

---

## Bonus Section: Two-Level Memory Hierarchy

### Objective
Optimize performance using a two-level memory system.

### Memory Specifications

**Level 1 Memory (SRAM - Fast):**
- Size: 1 KB
- Speed: 1 clock cycle read access
- Models high-speed on-chip SRAM
- Implement single-cycle read operation

**Level 2 Memory (DRAM - Slow):**
- Same as previous implementation
- Speed: 4 clock cycles read access
- Simulate slowness using a counter or FSM

### Implementation Strategy
- Transfer required weights from Level 2 to Level 1
- Perform computations using fast Level 1 memory
- Measure performance improvement

---

<!-- ## Submission Guidelines

**File Format:** ZIP file named `HW3_StudentNumber_Fullname.zip`

**Structure:**
- Part 1: Software implementation (training code, weights, report)
- Part 2: Three hardware implementations (WS, OS, IS)
- Part 3: Evaluation results and analysis
- Bonus: Two-level memory implementation (if completed)

**Report Requirements:**
- Use the provided report template (mandatory)
- Reports outside the template will not be reviewed

**Extensions:**
- Contact TA on Telegram before requesting extension
- Students have 4 coupon days available

--- -->

## Key Learning Outcomes

- Understanding CNN operations and LeNet architecture
- Implementing systolic array architectures
- Exploring different dataflow strategies (WS, OS, IS)
- Hardware performance evaluation (throughput, power)
- Memory hierarchy design and optimization
- Comparing software vs. hardware implementations

---

## References

- Reference textbook Section 5.6: Dataflows
- Reference textbook Table 5.1: Dataflow architectures
- Reference textbook Figure 5.16: NeuFlow architecture
- Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

---
