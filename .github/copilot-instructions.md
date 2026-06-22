# Micrograd Copilot Instructions

## Project Overview

**Micrograd** is a minimal educational autograd engine that implements reverse-mode automatic differentiation (backpropagation) over dynamically-built computational graphs. It consists of:
- **`engine.py`**: The `Value` class implementing scalar autograd (~120 lines)
- **`nn.py`**: PyTorch-like neural network building blocks (~60 lines)

This is **not** a production library—it's designed to teach how automatic differentiation works under the hood.

## Architecture

### The Autograd Engine (`engine.py`: `Value` class)

**Core Design Pattern**: Each `Value` node stores:
- `data`: The scalar numeric result
- `grad`: Accumulated gradient (initialized to 0)
- `_backward`: A lambda capturing the chain rule operation for this node
- `_prev`: Set of immediate children (used for topological sorting)
- `_op`: String label of operation (for debugging/visualization)

**Key Design Choice**: Operations are **lazy**—the backward pass is only executed when `backward()` is explicitly called. This chains rule calculations are deferred until needed.

**Operations Implemented**:
- Basic arithmetic: `+`, `-`, `*`, `/`
- Power: `**`
- Non-linearity: `relu()`
- Reverse ops: `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__` (for `2 + Value(3)`)

**Backward Pass Algorithm** (lines 59-65):
1. Build topological order via DFS (visits children before parents)
2. Iterate in reverse topological order
3. Apply stored `_backward()` lambda to chain gradients

### Neural Network Module (`nn.py`)

**Hierarchy**:
- `Module`: Base class with `parameters()` and `zero_grad()` methods
- `Neuron`: Single neuron with weights, bias, optional ReLU activation
- `Layer`: Collection of neurons operating on same input
- `MLP`: Multi-layer perceptron; chains layers, final layer always linear

**Pattern**: All network components inherit `Module` to enable `model.parameters()` for optimization loops.

## Testing Strategy

Tests in `test/test_engine.py` **validate against PyTorch**:
- Forward pass equivalence
- Gradient correctness (numerical tolerance ±1e-6)
- See line 24 & 54 for typical test patterns

This is the source of truth for numerical correctness.

## Common Workflows

### Adding Operations to `Value`
1. Define forward computation in the operation method (e.g., `__add__`)
2. Create `_backward()` lambda implementing the partial derivative
3. Add reverse operation (e.g., `__radd__` for scalar-first addition)
4. Add corresponding PyTorch test to `test_engine.py`

Example from `__mul__`:
```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad += other.data * out.grad  # ∂(self*other)/∂self = other
        other.grad += self.data * out.grad   # ∂(self*other)/∂other = self
    out._backward = _backward
    return out
```

### Building a Model
```python
from micrograd import nn
model = nn.MLP(nin=2, nouts=[16, 16, 1])  # 2-layer net, hidden size 16
```
MLP constructor takes `nouts` (list of layer sizes), **not** `out` or total layers.

### Training Pattern (from `demo.ipynb`)
1. Forward pass: `loss = criterion(model(X), y)`
2. Backward: `loss.backward()`
3. Manual SGD update: iterate `for p in model.parameters()` and update `p.data -= lr * p.grad`
4. `model.zero_grad()` before next iteration

## Design Conventions

**Why scalar-valued only?** Vectorized operations are intentionally avoided to expose the fundamental autograd mechanics. Each neuron output is computed as `sum(w*x) + b`, breaking matrix operations into individual scalar multiplications and additions.

**Implicit Type Coercion**: All ops convert Python floats to `Value` objects automatically (line 9, 18, 33 etc.). This enables expressions like `2 * Value(3) + 1`.

**No Explicit Parameter Initialization**: Weights initialized via `random.uniform(-1, 1)` in `Neuron.__init__`. Bias always starts at 0.

## File Structure & Key Lines

| Component | File | Key Lines |
|-----------|------|-----------|
| `Value` class | `engine.py` | 1-95 (ops), 59-65 (backward pass) |
| Network hierarchy | `nn.py` | 3-15 (Module), 17-26 (Neuron), 28-39 (Layer), 41-49 (MLP) |
| Tests | `test/test_engine.py` | Compare micrograd vs PyTorch side-by-side |
| Demo | `demo.ipynb` | Full training example on moon dataset |
| Visualization | `trace_graph.ipynb` | Graphviz rendering of computation graphs |

## Integration Points

- No external dependencies except `torch` for testing
- Relies on Python's built-in `set` for topological sorting, `random` for weight init
- Designed as **pure Python**—no NumPy, no autograd library dependencies

## Development Notes

- **Gradients accumulate**: `self.grad +=` not `self.grad =`. Important for multi-path computation graphs.
- **Backward called once per loss**: Typically one backward pass per training iteration.
- **Educational focus**: Code prioritizes clarity over performance. E.g., topological sort is O(n) but happens in Python for legibility.
