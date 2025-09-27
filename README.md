# Oven-Tensor

A PyTorch-style tensor library with GPU acceleration using CUDA kernels compiled by oven-compiler.

## Features

- **PyTorch-like Interface**: Familiar tensor operations with `.to()`, `.cpu()`, `.gpu()` methods
- **Automatic Kernel Compilation**: Python kernels compiled to PTX using oven-compiler
- **Smart Caching**: Compiled kernels cached for fast subsequent loads
- **CPU/GPU Hybrid**: Seamless switching between NumPy (CPU) and CUDA (GPU)
- **Extensible**: Easy to add custom kernels

## Installation

```bash
pip install oven-tensor
```

**Requirements:**
- Python 3.7+
- CUDA-capable GPU
- [oven-compiler](https://github.com/oven-lang/oven) in PATH
- PyCUDA

## Quick Start

```python
import oven_tensor as ot

# Create tensors
x = ot.tensor([1.0, 2.0, 3.0, 4.0])
y = ot.tensor([2.0, 3.0, 4.0, 5.0])

# CPU operations (NumPy)
z_cpu = x + y
print(z_cpu)  # Tensor([3. 5. 7. 9.], device=cpu)

# GPU operations (CUDA)
x_gpu = x.gpu()
y_gpu = y.gpu()
z_gpu = x_gpu + y_gpu
print(z_gpu.cpu())  # Tensor([3. 5. 7. 9.], device=cpu)

# Matrix multiplication
A = ot.tensor([[1.0, 2.0], [3.0, 4.0]])
B = ot.tensor([[5.0, 6.0], [7.0, 8.0]])
C = A @ B
print(C)  # Tensor([[19. 22.], [43. 50.]], device=cpu)
```

## Operations

### Tensor Creation
```python
ot.tensor([1, 2, 3])      # From data
ot.zeros((2, 3))          # Zero tensor
ot.ones((2, 3))           # Ones tensor
ot.randn((2, 3))          # Random normal
```

### Unary Operations
```python
x.sigmoid(), x.exp(), x.sqrt(), x.abs()
x.sin(), x.cos(), x.log(), x.tanh()
```

### Binary Operations
```python
x + y, x - y, x * y, x / y, x ** y, x % y
```

### Matrix Operations
```python
# Matrix multiplication
A = ot.tensor([[1, 2], [3, 4]])
B = ot.tensor([[5, 6], [7, 8]])

C = A.matmul(B)           # Method call
C = A @ B                 # @ operator
C = ot.matmul(A, B)       # Function call
```

### Device Management
```python
x.gpu()                   # Move to GPU
x.cpu()                   # Move to CPU
x.to(ot.device('gpu'))    # Explicit device
```

## Cache Management

```bash
# Command-line tool
oven-tensor-cache list    # List functions
oven-tensor-cache clear   # Clear cache
oven-tensor-cache info    # Show cache info
```

```python
# Python API
ot.clear_kernel_cache()
ot.reload_kernels()
ot.list_available_functions()
```

## Custom Kernels

Add kernels in `oven_tensor/kernels/`:

```python
# my_kernel.py
import oven.language as ol

def my_function(x_ptr: ol.ptr, y_ptr: ol.ptr):
    idx = ol.get_global_id()
    x_val = ol.load(x_ptr, idx)
    y_val = x_val * 2.0 + 1.0
    ol.store(y_val, y_ptr, idx)
```

## Testing

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
pytest tests/ -m "not gpu"      # Skip GPU tests
pytest tests/ -m "not slow"     # Skip slow tests
pytest tests/ --cov=oven_tensor # With coverage

# Run specific test files
pytest tests/test_tensor_basic.py
pytest tests/test_kernel_cache.py
```

## License

MIT License