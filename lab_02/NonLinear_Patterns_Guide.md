# Modeling Non-Linear Patterns with Activation Functions

A comprehensive guide to Lab 02: Building neural networks that can learn curved patterns using ReLU activation and data normalization.

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Non-Linear Data](#the-problem-non-linear-data)
3. [Data Normalization (Deep Dive)](#data-normalization-deep-dive)
   - [What is Normalization?](#what-is-normalization)
   - [Z-Score Standardization](#z-score-standardization)
   - [Step-by-Step Calculation](#step-by-step-calculation)
   - [Why Normalize?](#why-normalize)
   - [Visual Impact](#visual-impact)
4. [Building the Non-Linear Model](#building-the-non-linear-model)
   - [Architecture](#architecture)
   - [How ReLU Creates Curves](#how-relu-creates-curves)
5. [Training Process](#training-process)
6. [De-Normalization (Deep Dive)](#de-normalization-deep-dive)
   - [What is De-Normalization?](#what-is-de-normalization)
   - [The Mathematical Inverse](#the-mathematical-inverse)
   - [Step-by-Step Calculation](#step-by-step-de-normalization-calculation)
   - [Why De-Normalization is Critical](#why-de-normalization-is-critical)
7. [Complete Prediction Pipeline](#complete-prediction-pipeline)
8. [Helper Functions](#helper-functions)
9. [Summary](#summary)

---

## Overview

| Aspect | Lab 01 (Linear) | Lab 02 (Non-Linear) |
|--------|-----------------|---------------------|
| **Data** | Bike only | Bike + Car |
| **Pattern** | Straight line | Curve |
| **Model** | `Linear(1,1)` | `Linear(1,3) → ReLU → Linear(3,1)` |
| **Parameters** | 2 | 10 |
| **Normalization** | None | Z-score standardization |
| **Epochs** | 500 | 3000 |

---

## The Problem: Non-Linear Data

### The Dataset

The combined bike and car delivery data contains 39 data points:

```python
distances = [1.0, 1.5, 2.0, 2.5, 3.0, ..., 19.0, 19.5, 20.0]  # miles
times     = [6.96, 9.67, 12.11, 14.56, 16.77, ..., 90.73, 90.39, 92.98]  # minutes
```

### Why It's Non-Linear

The data exhibits two distinct behaviors:

| Distance Range | Vehicle | Time Pattern |
|----------------|---------|--------------|
| 1-3 miles | Bike | Steep increase (~5 min/mile) |
| 3-20 miles | Car | Gradual increase (traffic → highway) |

```
Time (minutes)
  100│                          ___________
     │                       __/
   80│                    __/
     │                  _/
   60│                _/
     │              _/
   40│            /
     │          /
   20│        /
     │      /
    0│____/________________________________
      0    5    10    15    20    Distance (miles)

      ←─ Bike ─→←────── Car ──────→
```

### Why Linear Models Fail

A linear model can only produce: `Time = W × Distance + B`

No matter what values W and B take, this is always a straight line. It cannot capture the curve where bike deliveries transition to car deliveries.

---

## Data Normalization (Deep Dive)

### What is Normalization?

**Normalization** is the process of transforming data to a standard scale without distorting differences in ranges of values. It's a critical preprocessing step that:

1. Brings all features to comparable scales
2. Helps optimization algorithms converge faster
3. Prevents features with large values from dominating learning

### Z-Score Standardization

The normalization technique used in this lab is **Z-score standardization** (also called **standard scaling**):

```
x_normalized = (x - μ) / σ

Where:
  x           = original value
  μ (mu)      = mean of all values
  σ (sigma)   = standard deviation of all values
  x_normalized = transformed value
```

**Result**: Data is transformed to have:
- **Mean = 0** (centered at zero)
- **Standard Deviation = 1** (unit variance)

### Step-by-Step Calculation

#### Step 1: Calculate Mean (μ)

```
μ = (1/n) × Σxᵢ

For distances [1.0, 1.5, 2.0, ..., 20.0]:
  Sum = 1.0 + 1.5 + 2.0 + ... + 20.0 = 409.5
  n = 39
  μ_distances = 409.5 / 39 = 10.5 miles
```

#### Step 2: Calculate Standard Deviation (σ)

```
σ = √[(1/n) × Σ(xᵢ - μ)²]

For distances:
  Variance = [(1-10.5)² + (1.5-10.5)² + ... + (20-10.5)²] / 39
           = [90.25 + 81.00 + ... + 90.25] / 39
           ≈ 33.25

  σ_distances = √33.25 ≈ 5.77 miles
```

#### Step 3: Apply Normalization

```
For each distance value:
  distance_norm = (distance - 10.5) / 5.77

Examples:
  1.0 miles  → (1.0 - 10.5) / 5.77  = -1.65
  10.5 miles → (10.5 - 10.5) / 5.77 =  0.00  (mean becomes 0)
  20.0 miles → (20.0 - 10.5) / 5.77 = +1.65
```

#### PyTorch Implementation

```python
# Calculate statistics
distances_mean = distances.mean()  # tensor(10.5)
distances_std = distances.std()    # tensor(5.77)
times_mean = times.mean()          # tensor(58.7)
times_std = times.std()            # tensor(28.5)

# Apply normalization
distances_norm = (distances - distances_mean) / distances_std
times_norm = (times - times_mean) / times_std
```

### Why Normalize?

#### Problem 1: Gradient Magnitude Imbalance

Without normalization, different features have vastly different scales:

```
Raw data:
  distances: 1.0 to 20.0 (range of 19)
  times: 6.96 to 92.98 (range of 86)

Gradients for larger values dominate the learning process!
```

#### Problem 2: Elongated Loss Surface

```
Without Normalization:              With Normalization:

Loss                                Loss
  │    ╱╲                             │      ∩
  │   ╱  ╲   ← Narrow valley          │     ╱ ╲   ← Spherical
  │  ╱    ╲    (slow zigzag)          │    ╱   ╲    (direct path)
  │ ╱      ╲                          │   ╱     ╲
  │╱________╲                         │__╱_______╲__
      w1                                   w1

Gradient descent takes                Gradient descent takes
many steps, zigzagging               direct path to minimum
```

#### Problem 3: Learning Rate Sensitivity

```
Without normalization:
  - Large features need small learning rate
  - Small features need large learning rate
  - One learning rate cannot satisfy both!

With normalization:
  - All features on same scale
  - Single learning rate works well for all
```

#### Summary of Benefits

| Without Normalization | With Normalization |
|----------------------|-------------------|
| Slow convergence | Fast convergence |
| Unstable gradients | Stable gradients |
| Learning rate hard to tune | Easier hyperparameter tuning |
| Features compete unfairly | All features contribute equally |
| May not converge at all | Reliable convergence |

### Visual Impact

```
BEFORE NORMALIZATION:

Distance (miles)                Time (minutes)
    20 │        •                   100│            •••
       │      •                        │         •••
    15 │    •                       80│       ••
       │   •                          │     ••
    10 │  •                         60│   ••
       │ •                            │  •
     5 │•                           40│ •
       │                              │•
     0 │___________                 20│___________

       Scale: 0-20                    Scale: 0-100


AFTER NORMALIZATION:

Normalized Distance              Normalized Time
   1.5 │        •                  1.0│            •••
       │      •                       │         •••
   0.5 │    •                      0.0│       ••
       │   •                          │     ••
   0.0 │  •     ← Mean is now 0   -0.5│   ••
       │ •                            │  •
  -0.5 │•                          -1.0│ •
       │                              │•
  -1.5 │___________                -2.0│___________

       Scale: -1.7 to +1.7            Scale: -2.0 to +1.0

Both scales are now comparable!
```

---

## Building the Non-Linear Model

### Architecture

```python
torch.manual_seed(27)  # For reproducibility

model = nn.Sequential(
    nn.Linear(1, 3),   # Layer 1: 1 input → 3 hidden neurons
    nn.ReLU(),         # Activation: introduces non-linearity
    nn.Linear(3, 1)    # Layer 2: 3 hidden → 1 output
)
```

### Visual Architecture

```
                         HIDDEN LAYER
                    ┌────────────────────┐
                    │                    │
                    │   ┌────┐           │
                    │ ┌→│ N1 │→ReLU──┐   │
                    │ │ └────┘       │   │
    INPUT           │ │              │   │         OUTPUT
  (normalized       │ │ ┌────┐       │   │       (normalized
   distance)        │ ├→│ N2 │→ReLU──┼───┼──┐     time)
       │            │ │ └────┘       │   │  │        │
       │  ┌──────┐  │ │              │   │  │  ┌──────┐
       └─→│Linear│──┼─┤              ├───┼──┼─→│Linear│──→
          │(1→3) │  │ │ ┌────┐       │   │  │  │(3→1) │
          └──────┘  │ └→│ N3 │→ReLU──┘   │  │  └──────┘
                    │   └────┘           │  │
                    │                    │  │
                    └────────────────────┘  │
                                            │
                    Each neuron creates     Output combines
                    a "hinge" point         all hinges
```

### Parameter Count

```
Layer 1: nn.Linear(1, 3)
  - Weights: 1 × 3 = 3 parameters
  - Biases:  3 parameters
  - Total:   6 parameters

Layer 2: nn.Linear(3, 1)
  - Weights: 3 × 1 = 3 parameters
  - Biases:  1 parameter
  - Total:   4 parameters

TOTAL MODEL PARAMETERS: 10
```

### How ReLU Creates Curves

#### Single Neuron with ReLU

```
One neuron computes: output = ReLU(w × x + b) = max(0, w × x + b)

This creates a "hinge" at x = -b/w:

If w > 0:                    If w < 0:
  output                       output
    │      /                     │
    │     /                      │\
    │    /                       │ \
    │___/________                │__\________
       -b/w                         -b/w

    "Hinge" opens right          "Hinge" opens left
```

#### Combining Multiple ReLU Neurons

```
With 3 neurons, you get 3 hinges at different positions:

Neuron 1: hinge at x₁    Neuron 2: hinge at x₂    Neuron 3: hinge at x₃
     │    /                   │      /                  │        /
     │   /                    │     /                   │       /
     │__/___                  │____/___                 │______/___
        x₁                        x₂                        x₃

Output layer combines them with learned weights (v₁, v₂, v₃):

Final output = v₁ × ReLU₁ + v₂ × ReLU₂ + v₃ × ReLU₃ + bias

Result: A piecewise linear function that approximates curves!

     │          ___----
     │       __/
     │    __/
     │  _/
     │ /
     │/_______________
```

#### Mathematical Representation

```
For input x (normalized distance):

h₁ = max(0, w₁x + b₁)   ← First hidden neuron
h₂ = max(0, w₂x + b₂)   ← Second hidden neuron
h₃ = max(0, w₃x + b₃)   ← Third hidden neuron

output = v₁h₁ + v₂h₂ + v₃h₃ + b_out

Where:
  w₁, w₂, w₃ = input weights (learned)
  b₁, b₂, b₃ = hidden biases (learned)
  v₁, v₂, v₃ = output weights (learned)
  b_out = output bias (learned)
```

---

## Training Process

### Setup

```python
loss_function = nn.MSELoss()                      # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent
```

### Training Loop

```python
for epoch in range(3000):
    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Forward pass (using NORMALIZED data)
    outputs = model(distances_norm)

    # 3. Compute loss (against NORMALIZED targets)
    loss = loss_function(outputs, times_norm)

    # 4. Backward pass
    loss.backward()

    # 5. Update parameters
    optimizer.step()
```

### Why More Epochs?

| Lab 01 (Linear) | Lab 02 (Non-Linear) |
|-----------------|---------------------|
| 500 epochs | 3000 epochs |
| 2 parameters to learn | 10 parameters to learn |
| Simple straight line | Complex curve with hinges |
| Fast convergence | Slower convergence |

### Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING ITERATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   distances_norm ──────┐                                         │
│   (normalized input)   │                                         │
│                        ↓                                         │
│                  ┌──────────┐                                    │
│                  │  MODEL   │                                    │
│                  │ (forward)│                                    │
│                  └────┬─────┘                                    │
│                       │                                          │
│                       ↓                                          │
│               outputs (normalized)                               │
│                       │                                          │
│                       ↓                                          │
│              ┌─────────────────┐                                 │
│              │   MSE LOSS      │←── times_norm                   │
│              │ (outputs vs     │    (normalized target)          │
│              │  times_norm)    │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│                       ↓                                          │
│              loss.backward()                                     │
│              (compute gradients)                                 │
│                       │                                          │
│                       ↓                                          │
│              optimizer.step()                                    │
│              (update weights)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## De-Normalization (Deep Dive)

### What is De-Normalization?

**De-normalization** (also called **inverse transformation** or **rescaling**) is the process of converting normalized predictions back to their original scale. This is essential because:

1. The model was trained on normalized data
2. The model outputs predictions in normalized scale
3. Users need predictions in real-world units (minutes, not z-scores)

### The Mathematical Inverse

Normalization formula:
```
x_normalized = (x - μ) / σ
```

To reverse this, solve for x:
```
x_normalized = (x - μ) / σ
x_normalized × σ = x - μ
x = x_normalized × σ + μ
```

**De-normalization formula:**
```
x_original = x_normalized × σ + μ

Where:
  x_normalized = model's output (in normalized scale)
  σ            = standard deviation (from training data)
  μ            = mean (from training data)
  x_original   = prediction in real-world units
```

### Step-by-Step De-Normalization Calculation

#### Given Values (from training data)

```python
times_mean = 58.7    # μ for times
times_std = 28.5     # σ for times
```

#### Example: Model Predicts 0.23 (normalized)

```
Step 1: Identify the normalized prediction
  predicted_norm = 0.23

Step 2: Apply de-normalization formula
  predicted_actual = predicted_norm × times_std + times_mean
                   = 0.23 × 28.5 + 58.7
                   = 6.555 + 58.7
                   = 65.26 minutes

Step 3: Interpret the result
  The model predicts a delivery time of approximately 65 minutes.
```

#### Multiple Examples

| Model Output (normalized) | Calculation | Actual Time (minutes) |
|---------------------------|-------------|----------------------|
| -2.0 | (-2.0 × 28.5) + 58.7 | 1.7 |
| -1.0 | (-1.0 × 28.5) + 58.7 | 30.2 |
| 0.0 | (0.0 × 28.5) + 58.7 | 58.7 (mean) |
| +0.5 | (+0.5 × 28.5) + 58.7 | 72.95 |
| +1.0 | (+1.0 × 28.5) + 58.7 | 87.2 |

### Why De-Normalization is Critical

#### Without De-Normalization

```python
predicted_norm = model(input_norm)
print(f"Delivery time: {predicted_norm.item()}")

Output: "Delivery time: 0.23"

User: "What does 0.23 mean? 0.23 minutes? 0.23 hours?"
       This is meaningless without context!
```

#### With De-Normalization

```python
predicted_norm = model(input_norm)
predicted_actual = (predicted_norm * times_std) + times_mean
print(f"Delivery time: {predicted_actual.item():.1f} minutes")

Output: "Delivery time: 65.3 minutes"

User: "Clear! I understand this will take about 65 minutes."
```

### Important Rules for De-Normalization

```
┌─────────────────────────────────────────────────────────────────┐
│                    DE-NORMALIZATION RULES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ALWAYS use the SAME μ and σ from training data               │
│     - Never recalculate from test/prediction data                │
│     - Store these values after training                          │
│                                                                  │
│  2. De-normalize OUTPUTS, not inputs                             │
│     - Input: normalize using input statistics                    │
│     - Output: de-normalize using output statistics               │
│                                                                  │
│  3. Order matters                                                │
│     - Normalization: (x - μ) / σ                                 │
│     - De-normalization: (x × σ) + μ                              │
│     - These are EXACT inverses                                   │
│                                                                  │
│  4. Match the feature                                            │
│     - distances_norm uses distances_mean, distances_std          │
│     - times_norm uses times_mean, times_std                      │
│     - Never mix them!                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### PyTorch Implementation

```python
# Store statistics after computing them (during training setup)
distances_mean = distances.mean()  # Save for later
distances_std = distances.std()    # Save for later
times_mean = times.mean()          # Save for later
times_std = times.std()            # Save for later

# During prediction
with torch.no_grad():
    # Normalize input
    new_distance = torch.tensor([[5.1]])
    new_distance_norm = (new_distance - distances_mean) / distances_std

    # Get normalized prediction
    predicted_norm = model(new_distance_norm)

    # De-normalize output
    predicted_actual = (predicted_norm * times_std) + times_mean

    print(f"Predicted time: {predicted_actual.item():.1f} minutes")
```

---

## Complete Prediction Pipeline

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPLETE PREDICTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   USER INPUT                                                     │
│   distance = 5.1 miles                                           │
│        │                                                         │
│        ↓                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              STEP 1: NORMALIZE INPUT                     │   │
│   │                                                          │   │
│   │   distance_norm = (5.1 - 10.5) / 5.77                    │   │
│   │                 = -5.4 / 5.77                            │   │
│   │                 = -0.936                                 │   │
│   │                                                          │   │
│   │   Why? Model only understands normalized scale           │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                         │
│        ↓                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              STEP 2: MODEL PREDICTION                    │   │
│   │                                                          │   │
│   │   input: -0.936 (normalized distance)                    │   │
│   │      │                                                   │   │
│   │      ↓                                                   │   │
│   │   Linear(1→3) → ReLU → Linear(3→1)                       │   │
│   │      │                                                   │   │
│   │      ↓                                                   │   │
│   │   output: -0.25 (normalized time)                        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                         │
│        ↓                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              STEP 3: DE-NORMALIZE OUTPUT                 │   │
│   │                                                          │   │
│   │   predicted_actual = (-0.25 × 28.5) + 58.7               │   │
│   │                    = -7.125 + 58.7                       │   │
│   │                    = 51.6 minutes                        │   │
│   │                                                          │   │
│   │   Why? User needs answer in real-world units             │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                         │
│        ↓                                                         │
│   USER OUTPUT                                                    │
│   "Predicted delivery time: 51.6 minutes"                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Code Example

```python
def predict_delivery_time(model, distance_miles,
                          distances_mean, distances_std,
                          times_mean, times_std):
    """
    Make a delivery time prediction for a given distance.

    Args:
        model: Trained PyTorch model
        distance_miles: Distance in miles (real-world units)
        distances_mean: Mean of training distances
        distances_std: Std of training distances
        times_mean: Mean of training times
        times_std: Std of training times

    Returns:
        Predicted delivery time in minutes
    """
    with torch.no_grad():
        # Step 1: Create tensor and normalize input
        distance_tensor = torch.tensor([[distance_miles]], dtype=torch.float32)
        distance_norm = (distance_tensor - distances_mean) / distances_std

        # Step 2: Get model prediction (in normalized scale)
        predicted_norm = model(distance_norm)

        # Step 3: De-normalize to get actual minutes
        predicted_minutes = (predicted_norm * times_std) + times_mean

        return predicted_minutes.item()

# Usage
distance = 5.1  # miles
time = predict_delivery_time(model, distance,
                             distances_mean, distances_std,
                             times_mean, times_std)
print(f"A {distance}-mile delivery will take approximately {time:.1f} minutes")
```

---

## Helper Functions

### `plot_data(distances, times, normalize=False)`

**Purpose**: Visualize the raw data points

```python
def plot_data(distances, times, normalize=False):
    plt.figure(figsize=(8, 6))
    plt.plot(distances.numpy(), times.numpy(),
             color='orange', marker='o', linestyle='none',
             label='Actual Delivery Times')

    if normalize:
        plt.title('Normalized Delivery Data (Bikes & Cars)')
        plt.xlabel('Normalized Distance')
        plt.ylabel('Normalized Time')
    else:
        plt.title('Delivery Data (Bikes & Cars)')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Time (minutes)')

    plt.legend()
    plt.grid(True)
    plt.show()
```

### `plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)`

**Purpose**: Show trained model predictions vs actual data (de-normalized)

```python
def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    model.eval()

    with torch.no_grad():
        # Get normalized predictions
        predicted_norm = model(distances_norm)

    # DE-NORMALIZE predictions to original scale
    predicted_times = (predicted_norm * times_std) + times_mean

    plt.figure(figsize=(8, 6))
    plt.plot(distances.numpy(), times.numpy(),
             color='orange', marker='o', linestyle='none',
             label='Actual Data')
    plt.plot(distances.numpy(), predicted_times.numpy(),
             color='green', label='Model Predictions')
    plt.title('Non-Linear Model Fit vs. Actual Data')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### `plot_training_progress(epoch, loss, model, distances_norm, times_norm)`

**Purpose**: Live visualization during training (shows learning progress)

```python
def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    clear_output(wait=True)  # Clear previous plot

    predicted_norm = model(distances_norm)

    # Convert to numpy for plotting
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()
    y_pred_plot = predicted_norm.detach().numpy()

    # Sort for smooth line
    sorted_indices = x_plot.argsort(axis=0).flatten()

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_plot, color='orange', marker='o',
             linestyle='none', label='Actual Data')
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices],
             color='green', label='Model Predictions')
    plt.title(f'Epoch: {epoch + 1} | Training Progress')
    plt.xlabel('Normalized Distance')
    plt.ylabel('Normalized Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    time.sleep(0.05)  # Brief pause for animation
```

---

## Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Non-linear data** | Patterns that curves, not straight lines |
| **ReLU activation** | `max(0, x)` - creates "hinges" for curve approximation |
| **Hidden layer** | Intermediate neurons that extract features |
| **Normalization** | `(x - μ) / σ` - scales data for stable training |
| **De-normalization** | `(x × σ) + μ` - converts predictions to real units |

### Normalization vs De-Normalization

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│   NORMALIZATION                    DE-NORMALIZATION               │
│   (Before Training/Prediction)     (After Prediction)             │
│                                                                   │
│   x_norm = (x - μ) / σ             x_actual = (x_norm × σ) + μ   │
│                                                                   │
│   • Applied to inputs              • Applied to outputs           │
│   • Uses input statistics          • Uses output statistics       │
│   • Prepares data for model        • Makes output interpretable   │
│                                                                   │
│   Example:                         Example:                       │
│   5.1 miles → -0.936               -0.25 → 51.6 minutes           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Complete Pipeline Summary

```
Raw Input (miles)
       │
       ↓
   NORMALIZE ──→ Model ──→ DENORMALIZE
   (x-μ)/σ                  (x×σ)+μ
       │                       │
       ↓                       ↓
Normalized Input    →    Actual Output (minutes)
```

### What You Learned

1. **Linear models have limits** - they can only fit straight lines
2. **ReLU enables curves** - each neuron adds a potential "bend"
3. **Normalization stabilizes training** - keeps gradients healthy
4. **De-normalization is essential** - converts outputs to useful units
5. **Store your statistics** - you need μ and σ for predictions
6. **More complexity = more training** - 3000 epochs vs 500
