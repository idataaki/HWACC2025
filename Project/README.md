# Hardware Accelerator Final Project: LeNet-5 CNN Accelerator with Systolic Arrays

## Faculty of Engineering and Computer Science - Shahid Beheshti University

### Academic Year: 2024-2025
**Release Date:** 1404/04/06 - **Phase 1 Due:** 1404/04/16 - **Phase 2 Due:** 1404/05/06

**Submission Format:** A `zip` file named `Fullname_StudentNumber_Project_FINAL.zip`

## Overview

Design and implement a hardware accelerator for **LeNet-5 CNN** using **systolic arrays with output-stationary dataflow**. The project combines software optimization techniques (pruning and quantization) with hardware acceleration to create an efficient inference engine for the **Fashion MNIST** dataset.

## Dataset

**Fashion MNIST**
- 70,000 grayscale images of clothing items (60K training, 10K testing)
- Each image: 28x28 pixels
- Zero-padded to 32x32 for hardware implementation
- 10 classes of fashion items

## Phase 1: Software Implementation and Optimization

### Step 1: Network Pruning
Implement two pruning methods with different sparsity ratios (25%, 50%, 70%, 90%, 95%, 97%, 99%):
- **Unstructured (Weight Pruning):** Zero out weights with smallest absolute values
- **Structured (Neuron/Filter Pruning):** Remove neurons/filters with lowest L2 norm

### Step 2: Post-Training Quantization
- Convert FP32 weights and activations to INT8 (8-bit integers)
- Use calibration with 100-500 training images
- Baseline: 90.53% accuracy (0.24 MB) → Quantized: 90.17% accuracy (0.06 MB)

### Step 3: Combined Optimization
- Apply pruning followed by quantization
- Evaluate final model size and accuracy

## Phase 2: Hardware Implementation

### Core Implementation: LeNet-5 Inference Engine
- **Architecture:** 5x5 systolic array with output-stationary dataflow
- **Data Precision:** INT8 (8-bit)
- **HDL:** Verilog/VHDL using Xilinx Vivado
- Implement all 5 layers of LeNet-5 (Conv1, Pool, Conv3, Pool, Conv5)

### Advanced Features

**1. Unstructured Sparsity Support**
- Implement zero-check unit to skip multiplications with zero operands
- Design bypass data path to avoid unnecessary MAC operations
- Compare power consumption and efficiency with/without sparsity support

**2. Two-Level Memory Hierarchy**
- **L1 Cache:** 1 KB SRAM, direct-mapped, 1-cycle latency
- **L2 Cache:** Block RAM (BRAM), 4-cycle latency
- Design cache management with LRU replacement policy
- **Weight Memory:** 64 KB read-only
- **Activation Memory:** 16 KB read-write

---

## Deliverables

### Final Report (use provided template - mandatory)
1. **Resource Utilization:** LUT, FF, DSP, BRAM consumption
2. **Power Consumption:** Vivado Power Analyzer results
3. **Inference Latency:** Execution time and delays
4. **Model Accuracy:** Classification accuracy on test set
5. **Speedup Analysis:** Compare vs. CPU (Google Colab) and ShiDianNao paper
6. **Power Comparison:** Analyze differences with ShiDianNao results

### Code and Simulations
- All HDL source code
- Synthesis and implementation results
- Simulation waveforms and testbench files

## LeNet-5 Architecture (32x32 input)

- **Input:** 32x32x1
- **Conv1:** 28x28x6 (5x5 kernel, 6 filters, stride 1)
- **S2 (Pool):** 14x14x6 (2x2, stride 2)
- **Conv3:** 10x10x16 (5x5 kernel, 16 filters, stride 1)
- **S4 (Pool):** 5x5x16 (2x2, stride 2)
- **Conv5:** 1x1x120 (5x5 kernel, 120 filters, stride 1)
- **F6 (FC):** 120 units
- **Output (FC):** 10 units

## Key Learning Outcomes

- CNN optimization through pruning and quantization
- Systolic array architecture and output-stationary dataflow
- Hardware implementation of neural network inference
- Memory hierarchy design and cache optimization
- Power and performance analysis in FPGA designs
- Comparison between software and hardware implementations

---

## Notes

- Reports not using the provided template will not be reviewed
- Contact TA via Telegram before requesting extensions
- 2-day coupon available after coordination
- Reference ShiDianNao accelerator for design guidance
- All synthesis must be completed successfully in Xilinx Vivado

---
