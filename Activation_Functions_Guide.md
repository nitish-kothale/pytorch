# Activation Functions in PyTorch - Comprehensive Guide

This document provides a complete reference for understanding and using activation functions in neural networks with PyTorch.

---

## Table of Contents

1. [What is an Activation Function?](#what-is-an-activation-function)
2. [Why Are Activation Functions Necessary?](#why-are-activation-functions-necessary)
3. [Detailed Analysis of Activation Functions](#detailed-analysis-of-activation-functions)
   - [ReLU](#1-relu-rectified-linear-unit)
   - [Sigmoid](#2-sigmoid-logistic-function)
   - [Tanh](#3-tanh-hyperbolic-tangent)
   - [Leaky ReLU](#4-leaky-relu)
   - [Softmax](#5-softmax)
   - [GELU](#6-gelu-gaussian-error-linear-unit)
4. [Gradient Flow and Backpropagation](#gradient-flow-and-backpropagation)
5. [Common Problems](#common-problems)
6. [Comparison Table](#comparison-table)
7. [Decision Flowchart](#decision-flowchart)
8. [Practical Code Examples](#practical-code-examples)
9. [Summary](#summary)

---

## What is an Activation Function?

An activation function is a mathematical function applied to the output of a neuron that introduces **non-linearity** into the network. Without activation functions, a neural network would just be a series of linear transformations, which can only learn linear patterns.

```
neuron_output = activation_function(weight × input + bias)
```

---

## Why Are Activation Functions Necessary?

### The Problem: Linear Models Can't Learn Curves

Consider a network with 3 linear layers (no activation):

```
Layer 1: h₁ = W₁x + b₁
Layer 2: h₂ = W₂h₁ + b₂
Layer 3: y  = W₃h₂ + b₃
```

Substitute step by step:

```
h₂ = W₂(W₁x + b₁) + b₂
   = W₂W₁x + W₂b₁ + b₂

y  = W₃(W₂W₁x + W₂b₁ + b₂) + b₃
   = W₃W₂W₁x + W₃W₂b₁ + W₃b₂ + b₃
   = (W₃W₂W₁)x + (W₃W₂b₁ + W₃b₂ + b₃)
   = W_effective × x + b_effective
```

**Result**: Three layers collapse into one linear transformation. You gain **zero** representational power by adding more linear layers.

### The Solution: Add Non-Linearity

```
Layer 1: h₁ = σ(W₁x + b₁)      ← σ is activation function
Layer 2: h₂ = σ(W₂h₁ + b₂)
Layer 3: y  = W₃h₂ + b₃
```

Now substitution gives:

```
y = W₃ × σ(W₂ × σ(W₁x + b₁) + b₂) + b₃
```

This **cannot** be simplified to a single linear equation. The nested non-linearities create a complex function that can approximate any continuous function (Universal Approximation Theorem).

### Visual Comparison

```
Without Activation (Linear only):

  Output│                          Can only draw
    ↑   │      /                   straight lines
        │     /
        │    /
        │   /
        │  /
        └─────────────→ Input


With Activation (Non-linear):

  Output│         ___----          Can draw curves,
    ↑   │      __/                 bends, and complex
        │    _/                    patterns!
        │   /
        │  /
        │_/
        └─────────────→ Input
```

---

## Detailed Analysis of Activation Functions

### 1. ReLU (Rectified Linear Unit)

**The most popular activation function for hidden layers.**

#### Definition

```
ReLU(x) = max(0, x) = { x  if x > 0
                      { 0  if x ≤ 0
```

#### Graph

```
output
  │      /
  │     /
  │    /
  │___/_________ input
     0
```

#### Derivative (Gradient)

```
d/dx ReLU(x) = { 1  if x > 0
              { 0  if x ≤ 0
              { undefined at x = 0 (PyTorch uses 0)
```

#### Why This Derivative Matters

During backpropagation:
- If `x > 0`: Gradient flows through unchanged (multiplied by 1)
- If `x ≤ 0`: Gradient is blocked (multiplied by 0)

```
Forward:  x = -3 → ReLU(-3) = 0
Backward: gradient × 0 = 0   ← No learning happens for this neuron!
```

#### The "Dying ReLU" Problem

```
Scenario:
  1. Neuron receives negative input for all training samples
  2. Output is always 0
  3. Gradient is always 0
  4. Weights never update
  5. Neuron is permanently "dead"

Example:
  If weights initialize such that W·x + b < 0 for all x in your data,
  that neuron will never activate and never learn.
```

#### PyTorch Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

# Method 1: nn.ReLU layer
relu_layer = nn.ReLU()
y = relu_layer(x)
print(y)  # tensor([0., 0., 0., 1., 2.])

# Method 2: Functional API
y = F.relu(x)

# Method 3: Direct tensor operation
y = torch.relu(x)

# Method 4: Manual implementation
y = torch.maximum(x, torch.zeros_like(x))

# Check gradients
y.sum().backward()
print(x.grad)  # tensor([0., 0., 0., 1., 1.])
```

#### Pros and Cons

| Pros | Cons |
|------|------|
| Computationally efficient (just a comparison) | "Dying ReLU" - neurons can get stuck at 0 |
| Doesn't saturate for positive values | Not zero-centered |
| Reduces vanishing gradient problem | |

---

### 2. Sigmoid (Logistic Function)

**Maps any value to a range between 0 and 1.**

#### Definition

```
σ(x) = 1 / (1 + e^(-x))

If x → +∞  → output → 1
If x → -∞  → output → 0
If x = 0   → output = 0.5
```

#### Graph

```
output
  1 │    ___-------
    │   /
0.5 │  /
    │ /
  0 │/_____________ input
        0
```

#### Derivative

```
d/dx σ(x) = σ(x) × (1 - σ(x))
```

This is elegant - the derivative is expressed in terms of the function itself.

#### Derivative Analysis

| x | σ(x) | σ'(x) = σ(x)(1-σ(x)) |
|---|------|----------------------|
| -10 | 0.00005 | 0.00005 |
| -5 | 0.0067 | 0.0066 |
| -2 | 0.119 | 0.105 |
| 0 | 0.5 | **0.25** (maximum) |
| 2 | 0.881 | 0.105 |
| 5 | 0.9933 | 0.0066 |
| 10 | 0.99995 | 0.00005 |

**Critical Observation**: Maximum gradient is only 0.25 (at x=0).

#### The Vanishing Gradient Problem

```
In a deep network with sigmoid activations:

Layer 10 gradient = gradient × σ'(...) × σ'(...) × ... × σ'(...)
                                 ↑         ↑              ↑
                              ≤ 0.25    ≤ 0.25         ≤ 0.25

After 10 layers: gradient ≤ 0.25^10 = 0.00000095

The gradient becomes so small that early layers barely learn!
```

#### Output Not Zero-Centered

```
Sigmoid output range: (0, 1)
Mean output: ~0.5 (not 0)

Problem for weight updates:
- If all inputs to a neuron are positive (from previous sigmoid)
- Gradients for all weights have the same sign
- Weights can only all increase or all decrease together
- This creates a "zig-zag" optimization path (slower convergence)
```

#### PyTorch Usage

```python
x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)

# Method 1: Layer
sigmoid_layer = nn.Sigmoid()
y = sigmoid_layer(x)
print(y)  # tensor([0.0067, 0.5000, 0.9933])

# Method 2: Function
y = torch.sigmoid(x)

# Check gradients
y.sum().backward()
print(x.grad)  # tensor([0.0066, 0.2500, 0.0066])
```

#### Use Cases

- Binary classification (output layer)
- Gates in LSTMs/GRUs
- When you need probability output (0-1)

---

### 3. Tanh (Hyperbolic Tangent)

**Maps any value to a range between -1 and 1.**

#### Definition

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = 2σ(2x) - 1      ← Scaled and shifted sigmoid!

If x → +∞  → output → +1
If x → -∞  → output → -1
If x = 0   → output = 0
```

#### Graph

```
output
 +1 │    ___-------
    │   /
  0 │--/----------- input
    │ /
 -1 │/
```

#### Derivative

```
d/dx tanh(x) = 1 - tanh²(x)
```

#### Comparison with Sigmoid

| Property | Sigmoid | Tanh |
|----------|---------|------|
| Output range | (0, 1) | (-1, 1) |
| Zero-centered | No | **Yes** |
| Max gradient | 0.25 | **1.0** |
| Saturates | Yes | Yes |

#### Why Zero-Centered Matters

```
Tanh output can be negative, zero, or positive.

For a neuron: y = Σ(w_i × x_i) + b

If inputs x_i can be positive or negative:
- Some weight gradients will be positive
- Some weight gradients will be negative
- Weights can update in different directions
- More direct path to optimal weights
```

#### PyTorch Usage

```python
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)

# As a layer
tanh_layer = nn.Tanh()
y = tanh_layer(x)

# As a function
y = torch.tanh(x)
print(y)  # tensor([-0.9640, 0.0000, 0.9640])

y.sum().backward()
print(x.grad)  # tensor([0.0707, 1.0000, 0.0707])
#                              ↑
#                     Max gradient = 1.0 at x=0
```

#### Use Cases

- Hidden layers in RNNs
- When zero-centered output is important

---

### 4. Leaky ReLU

**Fixes the "dying ReLU" problem by allowing small negative values.**

#### Definition

```
LeakyReLU(x) = { x        if x > 0
              { αx       if x ≤ 0     where α is small (default: 0.01)
```

#### Graph

```
output
  │      /
  │     /
  │    /
  │___/_________ input
  │ /
  │/  (small negative slope α)
```

#### Derivative

```
d/dx LeakyReLU(x) = { 1    if x > 0
                   { α    if x ≤ 0
```

#### Why It Fixes Dying ReLU

```
Standard ReLU:
  x = -3 → output = 0 → gradient = 0 → NO LEARNING

Leaky ReLU (α = 0.01):
  x = -3 → output = -0.03 → gradient = 0.01 → STILL LEARNING!
```

The small negative slope ensures gradients always flow, preventing neurons from dying.

#### Variants

```python
# Standard Leaky ReLU (fixed α)
nn.LeakyReLU(negative_slope=0.01)

# Parametric ReLU (α is learned during training)
nn.PReLU(num_parameters=1)  # One α for all channels
nn.PReLU(num_parameters=64) # Different α per channel

# Exponential Linear Unit (smooth version)
nn.ELU(alpha=1.0)
# ELU(x) = x if x > 0, else α(e^x - 1)

# Scaled ELU (self-normalizing networks)
nn.SELU()
```

#### PyTorch Usage

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
y = leaky_relu(x)
print(y)  # tensor([-0.0200, -0.0100, 0.0000, 1.0000, 2.0000])

y.sum().backward()
print(x.grad)  # tensor([0.0100, 0.0100, 0.0100, 1.0000, 1.0000])
#                        ↑       ↑       ↑
#                    Gradient = 0.01, not 0!
```

---

### 5. Softmax

**Converts a vector of values into probabilities that sum to 1.**

#### Definition

```
Softmax(x_i) = e^(x_i) / Σⱼ e^(x_j)
```

For a vector x = [x₁, x₂, ..., xₙ], each output is:

```
output_i = e^(x_i) / (e^(x_1) + e^(x_2) + ... + e^(x_n))
```

#### Properties

1. **All outputs are positive**: e^x > 0 for all x
2. **Outputs sum to 1**: Σ output_i = 1
3. **Preserves order**: If x_i > x_j, then output_i > output_j
4. **Amplifies differences**: Larger inputs get disproportionately larger outputs

#### Numerical Example

```
Input:  x = [2.0, 1.0, 0.1]

e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105
Sum   = 11.212

Softmax outputs:
  7.389 / 11.212 = 0.659  (65.9%)
  2.718 / 11.212 = 0.242  (24.2%)
  1.105 / 11.212 = 0.099  (9.9%)
                   ─────
                   1.000  (100%)
```

#### Numerical Stability Issue

```python
# Problem: Large values cause overflow
x = torch.tensor([1000.0, 1001.0, 1002.0])
# e^1000 = inf (overflow!)

# Solution: Subtract max value (mathematically equivalent)
x_stable = x - x.max()  # [-2, -1, 0]
# Now e^0 = 1, e^-1 = 0.368, e^-2 = 0.135 (safe!)
```

PyTorch's `F.softmax` handles this automatically.

#### Temperature Scaling

```python
# Standard softmax
F.softmax(x, dim=-1)

# With temperature T
F.softmax(x / T, dim=-1)

# T < 1: "Sharper" distribution (more confident)
# T > 1: "Softer" distribution (more uniform)
# T → 0: Approaches argmax (one-hot)
# T → ∞: Approaches uniform distribution
```

Example:
```
x = [2.0, 1.0, 0.0]

T=1.0: [0.659, 0.242, 0.099]  (standard)
T=0.5: [0.844, 0.114, 0.042]  (sharper)
T=2.0: [0.506, 0.307, 0.186]  (softer)
```

#### PyTorch Usage

```python
x = torch.tensor([[2.0, 1.0, 0.1]])  # Shape: [batch=1, classes=3]

# Must specify dimension!
probs = F.softmax(x, dim=1)  # Apply across classes (dim=1)
print(probs)      # tensor([[0.6590, 0.2424, 0.0986]])
print(probs.sum()) # tensor(1.0000)

# For numerical stability with cross-entropy loss,
# use log_softmax + nll_loss, or combined cross_entropy
log_probs = F.log_softmax(x, dim=1)
```

#### Use Case

- Multi-class classification (output layer)
- When you need probability distribution over classes

---

### 6. GELU (Gaussian Error Linear Unit)

**Used in modern transformers (BERT, GPT).**

#### Definition

```
GELU(x) = x × Φ(x)

Where Φ(x) is the CDF of standard normal distribution:
Φ(x) = P(X ≤ x) for X ~ N(0,1)
```

#### Approximation (used in practice)

```
GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

#### Intuition

```
GELU can be thought of as a "smooth" version of ReLU:

- For large positive x: Φ(x) ≈ 1, so GELU(x) ≈ x (like ReLU)
- For large negative x: Φ(x) ≈ 0, so GELU(x) ≈ 0 (like ReLU)
- For x near 0: Smooth transition (unlike ReLU's sharp corner)

The "stochastic" interpretation:
- GELU randomly zeros out inputs based on their value
- Larger values are more likely to be kept
- This acts as a form of regularization
```

#### Comparison with ReLU

| x | ReLU(x) | GELU(x) |
|---|---------|---------|
| -3.0 | 0.000 | -0.004 |
| -1.0 | 0.000 | -0.159 |
| -0.5 | 0.000 | -0.154 |
| 0.0 | 0.000 | 0.000 |
| 0.5 | 0.500 | 0.346 |
| 1.0 | 1.000 | 0.841 |
| 3.0 | 3.000 | 2.996 |

#### PyTorch Usage

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Exact computation
gelu_exact = nn.GELU(approximate='none')
print(gelu_exact(x))  # tensor([-0.0455, -0.1587, 0.0000, 0.8413, 1.9545])

# Tanh approximation (faster)
gelu_approx = nn.GELU(approximate='tanh')
print(gelu_approx(x))  # tensor([-0.0454, -0.1588, 0.0000, 0.8412, 1.9546])
```

#### Use Case

- State-of-the-art for transformer architectures (BERT, GPT, etc.)

---

## Gradient Flow and Backpropagation

Understanding how gradients flow through activation functions is crucial for training deep networks.

### Visualization

```
                    FORWARD PASS →
    ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────┐
x → │ Linear  │ → z │Activation│ → a │ Linear  │ → y │ Loss │ → L
    │ W₁, b₁  │     │   σ(z)   │     │ W₂, b₂  │     │      │
    └─────────┘     └──────────┘     └─────────┘     └──────┘

                    ← BACKWARD PASS
    ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────┐
    │ ∂L/∂W₁  │ ← ∂ │  σ'(z)   │ ← ∂ │ ∂L/∂W₂  │ ← ∂ │∂L/∂y │ ← 1
    │ ∂L/∂b₁  │     │ multiply │     │ ∂L/∂b₂  │     │      │
    └─────────┘     └──────────┘     └─────────┘     └──────┘
                         ↑
                    THIS IS WHERE
                    ACTIVATION CHOICE
                    MATTERS!
```

### Gradient Behavior by Activation

**If σ'(z) is:**
- **≈ 0** (sigmoid at extremes): Gradient vanishes, early layers don't learn
- **= 1** (ReLU for positive): Gradient flows perfectly
- **= 0** (ReLU for negative): Gradient blocked, neuron dies
- **= small constant** (Leaky ReLU): Gradient reduced but flows

---

## Common Problems

### 1. Vanishing Gradient

**Cause**: Activation derivatives are consistently < 1 (sigmoid, tanh at extremes)

**Effect**: Gradients shrink exponentially through layers

```
After 10 sigmoid layers: gradient ≤ 0.25^10 = 0.00000095
```

**Solutions**:
- Use ReLU or variants
- Use batch normalization
- Use residual connections (skip connections)

### 2. Dying ReLU

**Cause**: ReLU outputs 0 for all negative inputs

**Effect**: Neurons permanently stop learning

```
If W·x + b < 0 for all training samples → neuron is dead
```

**Solutions**:
- Use Leaky ReLU, PReLU, or ELU
- Careful weight initialization
- Lower learning rate

### 3. Exploding Gradient

**Cause**: Large weights combined with activation that doesn't bound outputs

**Effect**: Gradients grow exponentially, causing unstable training

**Solutions**:
- Gradient clipping
- Proper weight initialization
- Batch normalization

---

## Comparison Table

| Activation | Formula | Derivative | Range | Use Case | Gradient Issue |
|------------|---------|------------|-------|----------|----------------|
| **ReLU** | max(0, x) | 0 or 1 | [0, ∞) | Hidden layers | Dying neurons |
| **Leaky ReLU** | max(αx, x) | α or 1 | (-∞, ∞) | Hidden layers | None |
| **Sigmoid** | 1/(1+e⁻ˣ) | σ(1-σ) | (0, 1) | Binary output | Vanishing |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh² | (-1, 1) | RNNs | Vanishing |
| **Softmax** | eˣⁱ/Σeˣʲ | Complex | (0, 1) | Multi-class | N/A |
| **GELU** | x·Φ(x) | Φ(x)+x·φ(x) | ≈(-0.17, ∞) | Transformers | None |

---

## Decision Flowchart

```
                    What type of layer?
                           │
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
       Hidden          Output          Output
       Layer        (Binary class)  (Multi-class)
           │               │               │
           ↓               ↓               ↓
    ┌──────────────┐       │               │
    │ Modern arch? │       ↓               ↓
    │ (Transformer)│   Sigmoid         Softmax
    └──────┬───────┘
           │
     ┌─────┴─────┐
     ↓           ↓
    Yes          No
     │           │
     ↓           ↓
   GELU    ┌─────────────┐
           │ Dying ReLU  │
           │  a concern? │
           └─────┬───────┘
                 │
           ┌─────┴─────┐
           ↓           ↓
          Yes          No
           │           │
           ↓           ↓
      Leaky ReLU     ReLU
```

### Quick Reference

| Scenario | Recommended Activation |
|----------|------------------------|
| Hidden layers (default) | **ReLU** or **Leaky ReLU** |
| Binary classification output | **Sigmoid** |
| Multi-class classification output | **Softmax** |
| Regression output | **None** (linear) |
| RNN/LSTM hidden states | **Tanh** |
| Transformer models | **GELU** |

---

## Practical Code Examples

### Example 1: Comparing All Activations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create sample data
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

print("Input:", x.tolist())
print("-" * 50)
print(f"ReLU:       {F.relu(x).tolist()}")
print(f"Leaky ReLU: {F.leaky_relu(x, 0.01).tolist()}")
print(f"Sigmoid:    {[round(v, 4) for v in torch.sigmoid(x).tolist()]}")
print(f"Tanh:       {[round(v, 4) for v in torch.tanh(x).tolist()]}")
print(f"GELU:       {[round(v, 4) for v in F.gelu(x).tolist()]}")
```

Output:
```
Input: [-3.0, -1.0, 0.0, 1.0, 3.0]
--------------------------------------------------
ReLU:       [0.0, 0.0, 0.0, 1.0, 3.0]
Leaky ReLU: [-0.03, -0.01, 0.0, 1.0, 3.0]
Sigmoid:    [0.0474, 0.2689, 0.5, 0.7311, 0.9526]
Tanh:       [-0.9951, -0.7616, 0.0, 0.7616, 0.9951]
GELU:       [-0.0036, -0.1587, 0.0, 0.8413, 2.9964]
```

### Example 2: Building a Model with Activations

```python
import torch
import torch.nn as nn

class FlexibleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Choose activation function
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation, nn.ReLU())

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation on output layer
        return x

# Create models with different activations
for act in ['relu', 'leaky_relu', 'gelu', 'tanh']:
    model = FlexibleNet(10, 64, 2, activation=act)
    x = torch.randn(4, 10)
    output = model(x)
    print(f"{act.upper():12} output shape: {output.shape}")
```

### Example 3: Checking Gradients

```python
import torch
import torch.nn.functional as F

# Compare gradient flow through different activations
x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)

# ReLU gradients
y_relu = F.relu(x.clone().detach().requires_grad_(True))
y_relu.sum().backward()
print(f"ReLU gradients:       {x.grad}")

# Leaky ReLU gradients
x2 = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
y_leaky = F.leaky_relu(x2, 0.01)
y_leaky.sum().backward()
print(f"Leaky ReLU gradients: {x2.grad}")

# Sigmoid gradients
x3 = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
y_sigmoid = torch.sigmoid(x3)
y_sigmoid.sum().backward()
print(f"Sigmoid gradients:    {x3.grad}")
```

### Example 4: Multi-class Classification with Softmax

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simulated logits from a classifier (3 classes)
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3],
                       [0.1, 0.2, 3.0]])

# Convert to probabilities
probabilities = F.softmax(logits, dim=1)

print("Logits:")
print(logits)
print("\nProbabilities (each row sums to 1):")
print(probabilities)
print("\nRow sums:", probabilities.sum(dim=1))
print("\nPredicted classes:", probabilities.argmax(dim=1))
```

Output:
```
Logits:
tensor([[2.0000, 1.0000, 0.1000],
        [0.5000, 2.5000, 0.3000],
        [0.1000, 0.2000, 3.0000]])

Probabilities (each row sums to 1):
tensor([[0.6590, 0.2424, 0.0986],
        [0.1185, 0.8754, 0.0970],
        [0.0466, 0.0515, 0.9019]])

Row sums: tensor([1.0000, 1.0000, 1.0000])

Predicted classes: tensor([0, 1, 2])
```

---

## Summary

### Key Takeaways

1. **Activation functions introduce non-linearity** - allowing neural networks to learn complex patterns beyond straight lines

2. **Without activation functions**, stacking layers is mathematically equivalent to a single linear layer

3. **ReLU is the default choice** for hidden layers due to its simplicity and effectiveness

4. **Sigmoid and Softmax** are used for output layers when you need probability outputs

5. **Leaky ReLU** solves the dying neuron problem of standard ReLU

6. **GELU** is preferred in modern transformer architectures

7. **The choice of activation function** significantly impacts:
   - Training speed
   - Gradient flow
   - Model performance
   - Convergence stability

### Rule of Thumb

```
Hidden layers  → ReLU (or Leaky ReLU if dying neurons occur)
Binary output  → Sigmoid
Multi-class    → Softmax
Regression     → No activation (linear)
Transformers   → GELU
RNNs           → Tanh
```
