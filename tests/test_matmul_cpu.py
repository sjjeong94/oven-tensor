"""
Test matmul functionality (CPU only)
"""

import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Mock CUDA modules to avoid import issues
sys.modules["pycuda"] = MagicMock()
sys.modules["pycuda.driver"] = MagicMock()
sys.modules["pycuda.autoinit"] = MagicMock()

# Now import our tensor library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import oven_tensor as ot


class TestMatmulCPU:
    """Test matmul operations on CPU"""

    def test_matmul_2x2(self):
        """Test 2x2 matrix multiplication"""
        a = ot.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = ot.tensor([[5.0, 6.0], [7.0, 8.0]])

        # Test method call
        c = a.matmul(b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c._data, expected)

        # Test @ operator
        c2 = a @ b
        np.testing.assert_array_almost_equal(c2._data, expected)

        # Test package function
        c3 = ot.matmul(a, b)
        np.testing.assert_array_almost_equal(c3._data, expected)

    def test_matmul_different_sizes(self):
        """Test matrix multiplication with different sizes"""
        # Test 3x2 @ 2x4 = 3x4
        x = ot.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
        y = ot.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # 2x4

        z = x @ y  # Should be 3x4
        assert z.shape == (3, 4)

        # Compare with NumPy
        expected = np.dot(x.numpy(), y.numpy())
        np.testing.assert_array_almost_equal(z.numpy(), expected)

    def test_matmul_error_cases(self):
        """Test matmul error handling"""
        a = ot.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Test 1D tensor (should fail)
        b_1d = ot.tensor([1.0, 2.0])
        try:
            c = a @ b_1d
            assert False, "Should have raised RuntimeError for 1D tensor"
        except RuntimeError as e:
            assert "2D tensors" in str(e)

        # Test incompatible dimensions
        b_incompatible = ot.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )  # 3x3
        try:
            c = a @ b_incompatible  # 2x2 @ 3x3 should fail
            assert False, "Should have raised RuntimeError for incompatible dimensions"
        except RuntimeError as e:
            assert "incompatible" in str(e)

    def test_matmul_identity(self):
        """Test matmul with identity matrix"""
        a = ot.tensor([[1.0, 2.0], [3.0, 4.0]])
        identity = ot.tensor([[1.0, 0.0], [0.0, 1.0]])

        # a @ I = a
        result = a @ identity
        np.testing.assert_array_almost_equal(result.numpy(), a.numpy())

        # I @ a = a
        result2 = identity @ a
        np.testing.assert_array_almost_equal(result2.numpy(), a.numpy())


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
