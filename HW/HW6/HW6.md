# Hardware Accelerator Assignment 6: Energy Evaluation Tools

## Faculty of Engineering and Computer Science - Shahid Beheshti University

### Academic Year: 2024-2025 | Spring Semester
**Release Date:** 1404/03/07 - **Due Date:** 1404/03/17

**Instructor:** Dara Rahmati
**Assignment Designers:** Zahra Taki, Matin Firoozbakht, Alireza Taghavizadeh

**Submission Format:** A `zip` file containing all required components.

---

## Overview

This assignment introduces **energy evaluation tools** for hardware accelerator design. Energy optimization is critical for devices with limited energy resources, as it improves efficiency, operational stability, and battery life.

**Focus Areas:**
- Energy consumption analysis in neural network accelerators
- Using Timeloop-Accelergy for energy estimation
- Using MAESTRO for dataflow analysis
- Hardware modeling and optimization

---

## Introduction

### Why Energy Evaluation Matters

Hardware accelerators are widely used in energy-constrained devices. Energy evaluation tools play a vital role in:

- **Analyzing energy consumption** in different system components
- **Identifying high-power consumption** areas
- **Comparing different architectures**
- **Supporting design decisions**

This assignment focuses on two key tool suites:
1. **Timeloop-Accelergy**
2. **MAESTRO**

---

## Tool 1: Timeloop and Accelergy

### What are Timeloop and Accelergy?

**Timeloop** and **Accelergy** are open-source tools developed for designing and evaluating hardware accelerators for neural networks.

### Timeloop

**Purpose:** Modeling data mapping, scheduling, and memory hierarchy placement in accelerators.

**Key Features:**
- Models how neural network operations are mapped to hardware
- Analyzes execution efficiency on custom architectures
- Simulates memory hierarchy and data movement
- Optimizes scheduling and spatial placement

### Accelergy

**Purpose:** Accurate energy estimation for accelerator components.

**Key Features:**
- Estimates energy consumption of different hardware components
- Works in conjunction with Timeloop
- Provides detailed energy breakdowns
- Supports custom architecture specifications

### Combined Usage

These tools are typically used **together** to provide simultaneous evaluation of:
- **Execution time** (performance)
- **Energy consumption** (efficiency)

**Workflow:**
1. Define hardware architecture in Timeloop
2. Specify data mapping and scheduling
3. Use Accelergy to estimate energy for each component
4. Analyze results and optimize

---

## Tool 2: MAESTRO

### What is MAESTRO?

**MAESTRO** is a tool for rapid dataflow analysis in neural network accelerators and design space exploration.

### Key Features

- **Fast dataflow analysis:** Evaluates impact of different dataflow strategies
- **Design space exploration:** Simplifies search for optimal configurations
- **Resource analysis:** Examines memory usage, bandwidth, and energy
- **Comparison tool:** Compares different mapping strategies

### MAESTRO Workflow

**Inputs:**
1. **Neural network model** (can be generated using Keras/PyTorch)
2. **Hardware specifications**
3. **Mapping strategy** (dataflow)

**Outputs:**
- Latency metrics
- Bottleneck information
- NoC (Network-on-Chip) bandwidth requirements
- Activity counts (for energy estimation)
- Buffer size requirements
- Data reuse statistics

**Process:**
1. Tensor Analysis
2. Cluster Analysis
3. Reuse Analysis
4. Performance Analysis
5. Cost Analysis

---

## Hardware Accelerator: Simba

### Reference Architecture

For this assignment, use the **Simba** accelerator architecture as described in the research paper.

**Paper:** "Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture"

### Key Characteristics

- **Architecture:** Systolic array based
- **Modeling scope:** Single chip (not the full multi-chip module)
- **Components:** Processing elements, memory hierarchy, interconnects

### What to Model

Study the Simba paper to understand:
- Systolic array dimensions
- Memory hierarchy (Local RF, GLB, DRAM)
- Dataflow patterns
- Hardware specifications

**Important:** Model only **one chip** from the Simba architecture, not the entire multi-chip system.

---

## Part 1: Timeloop-Accelergy Tasks

### Task 1.1: Hardware Modeling with Timeloop

**Objective:** Model the systolic array portion of the Simba accelerator using Timeloop.

**Requirements:**

1. **Define Architecture:**
   - Systolic array dimensions
   - Input/output port specifications
   - Memory hierarchy (capacity and type of each level)
   - Dataflow pattern

2. **Specify Components:**
   - Computational elements (PEs)
   - Buffers at each level
   - Communication paths
   - Parameters for each component

3. **Configuration Details:**
   - Data mapping strategy
   - Execution scheduling
   - Memory hierarchy placement

**Deliverables:**
- Timeloop configuration files
- Explanation of each architectural choice
- Justification for parameters selected

### Task 1.2: Energy Estimation with Accelergy

**Objective:** Estimate energy consumption of the modeled architecture using Accelergy.

**Requirements:**

1. **Technology Parameters:**
   - Configure SRAM-based memory parameters
   - Specify technology node
   - Define component characteristics

2. **Energy Modeling:**
   - Local memory (RF) energy
   - Global buffer (GLB) energy
   - Compute unit energy
   - Data movement energy

3. **Analysis:**
   - Report energy consumption per component
   - Identify energy hotspots
   - Provide breakdown of total energy

**Deliverables:**
- Accelergy configuration files
- Energy consumption reports
- Analysis of energy distribution
- Recommendations for optimization

---

## Part 2: MAESTRO Tasks

### Installation

**Platforms Supported:**
- Linux operating system (recommended)
- Virtual machine (alternative)
- Docker image (tutorial version available)

**Prerequisites:**
- `libboost-all-dev`
- `scons`

**Installation Links:** Provided in references section

### Task 2.1: Understand Dataflows

**Objective:** Study and explain different dataflow strategies supported by MAESTRO.

**Instructions:**

1. Navigate to `dataflow/frontend/tools` in the MAESTRO repository
2. Identify all supported dataflow types
3. Briefly explain each dataflow

**Reference Paper:** Study the MAERI paper for detailed dataflow explanations.

**Deliverables:**
- List of all dataflows
- Concise explanation of each
- Comparison between different strategies

### Task 2.2: Dataflow Keywords

**Objective:** Explain the meaning of key dataflow specification keywords.

**Keywords to Define:**

1. **TemporalMap:** What does this specify?
2. **SpatialMap:** What does this specify?
3. **Cluster:** What does this specify?

**Deliverables:**
- Clear definition of each keyword
- Examples of usage
- Impact on hardware mapping

### Task 2.3: Neural Network Model Generation

**Objective:** Generate neural network models with different input sizes and compare outputs.

**Requirements:**

Using Keras or PyTorch, generate MobileNet_v2 models with:

**Model A:**
- Input size: `(3, 244, 244)`

**Model B:**
- Input size: `(3, 488, 488)`

**Model Type:**
- Use **MobileNet_v2** architecture

**Analysis:**
- Compare the main differences in final output
- Report changes in layer dimensions
- Analyze computational requirements
- Discuss memory implications

**Deliverables:**
- Model generation code (Keras/PyTorch)
- Exported model files
- Comparison report
- Analysis of differences

### Task 2.4: MAESTRO Output Analysis

**Objective:** Understand and analyze MAESTRO output parameters.

**Questions to Answer:**

1. What parameters does MAESTRO provide as accelerator analysis output?
2. Which parameters are most important in your opinion?
3. Which outputs have the greatest impact on input parameter selection?

**Analysis:**
- Identify all output metrics
- Rank them by importance
- Explain interdependencies
- Discuss optimization trade-offs

**Deliverables:**
- Complete list of output parameters
- Importance ranking with justification
- Discussion of parameter relationships

---

## Execution Guidelines

### General Notes

1. **Save All Outputs:** For each execution, save complete results with all parameters specified

2. **Question Order:** Questions are not necessarily in the order of tool learning/execution

3. **Parameter Comparison:** When comparing two parameters:
   - Keep all other parameters **constant**
   - Clearly state the constant parameter values
   - Can use default values (but document them)

4. **Complete Results:** Save outputs from all test cases after execution

### Best Practices

- **Documentation:** Record all configuration choices
- **Reproducibility:** Save exact parameter values used
- **Iteration:** Multiple runs may be needed for some analyses
- **Validation:** Cross-check results for consistency

---

## Submission Guidelines

**File Format:** ZIP file named `HW6_StudentNumber_Fullname.zip`

**Structure:**
- Separate subdirectory for each question/task
- All configuration files
- All output files
- All generated models
- Report (use provided template - **mandatory**)

**Report Contents:**
- Installation process and challenges
- Architecture modeling decisions
- Energy analysis results
- Dataflow explanations
- Model generation code and results
- MAESTRO output analysis
- Conclusions and insights

**Extensions:**
- Contact TA on Telegram before requesting extension
- Students have 4 coupon days available

**Important:**
- Reports must use the provided template
- Reports outside the template will **not be reviewed**

---

## Key Learning Outcomes

- Understanding energy evaluation in hardware accelerators
- Using Timeloop for architecture modeling
- Using Accelergy for energy estimation
- Analyzing dataflows with MAESTRO
- Modeling systolic array architectures
- Design space exploration techniques
- Energy-performance trade-off analysis
- Hardware-software co-design principles

---

## Important Concepts

### Memory Hierarchy

1. **Local RF (Register File):** Smallest, fastest memory close to PEs
2. **GLB (Global Buffer):** Shared memory for multiple PEs
3. **DRAM:** Main memory, largest capacity, higher latency

### Dataflow Strategies

Different ways to schedule and map neural network operations:
- **Weight Stationary (WS)**
- **Output Stationary (OS)**
- **Input Stationary (IS)**
- **Row Stationary (RS)**
- Others as discovered in MAESTRO

### Energy Components

- **Computation Energy:** Energy for arithmetic operations
- **Memory Access Energy:** Reading/writing to memory
- **Data Movement Energy:** Transferring data between memory levels
- **Leakage Energy:** Static power consumption

---

## References and Resources

### Papers

**Simba Paper:**
https://drive.google.com/file/d/1AKsFU1Usnv5DdXms2g6Ny09xamxjnkX9/view

**MAERI Tutorial:**
https://synergy.ece.gatech.edu/tools/maeri/maeri-tutorial-hpca-2019/

### Installation Guides

**Timeloop Installation:**
https://timeloop.csail.mit.edu/v4/installation

**MAESTRO Installation:**
http://maestro.ece.gatech.edu/docs/build/html/installation.html

### Repositories

**MAESTRO Project:**
https://github.com/maestro-project/maestro

---

## Tips for Success

### Timeloop-Accelergy

- Start with simple configurations and gradually increase complexity
- Validate models with known reference designs
- Pay attention to memory hierarchy bandwidth constraints
- Use provided examples as starting points

### MAESTRO

- Carefully read documentation for input format
- Start with provided example dataflows
- Understand the mapping specification syntax
- Compare results across different dataflows

### General

- **Docker recommended:** Easier setup, fewer dependency issues
- **Save intermediate results:** Avoid re-running long simulations
- **Document assumptions:** Critical for understanding results
- **Validate outputs:** Check if results make sense
- **Study examples:** Learn from provided reference implementations

---

## Common Pitfalls to Avoid

1. **Incorrect memory hierarchy specification:** Leads to invalid results
2. **Mismatched dimensions:** Between network layers and hardware
3. **Ignoring technology parameters:** Affects energy accuracy
4. **Not saving outputs:** Losing hours of simulation results
5. **Incomplete dataflow specification:** Causes errors or suboptimal mapping

---

## Evaluation Criteria

1. **Correctness of modeling** (Timeloop architecture)
2. **Energy analysis quality** (Accelergy results)
3. **Dataflow understanding** (MAESTRO concepts)
4. **Model generation** (Keras/PyTorch implementations)
5. **Output analysis depth** (MAESTRO metrics)
6. **Report quality** (clarity, completeness, insights)
7. **Documentation** (reproducibility, parameter choices)

---

## Additional Resources

### Recommended Reading

- Timeloop/Accelergy documentation and tutorials
- MAESTRO user guide and examples
- Simba paper (architecture details)
- MAERI paper (dataflow concepts)
- Related papers on systolic arrays and dataflows

### Tool Documentation

Both tools have extensive documentation:
- Example configurations
- API references
- Tutorial notebooks
- Design patterns

**Take time to study examples before starting your own implementations.**

---

Good luck!

**Instructor:** Rahmati
