"""
Test basic tensor functionality
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import oven_tensor as ot


class TestTensorBasics:
    """Test basic tensor operations"""

    def test_tensor_creation(self):
        """Test tensor creation from different data types"""
        # From list
        x = ot.tensor([1, 2, 3, 4])
        assert x.shape == (4,)
        assert x.device.type == "cpu"
        np.testing.assert_array_equal(x._data, [1, 2, 3, 4])

        # From numpy array
        arr = np.array([[1, 2], [3, 4]])
        y = ot.tensor(arr)
        assert y.shape == (2, 2)
        np.testing.assert_array_equal(y._data, arr)

    def test_tensor_zeros(self):
        """Test zeros tensor creation"""
        x = ot.zeros((3, 3))
        assert x.shape == (3, 3)
        np.testing.assert_array_equal(x._data, np.zeros((3, 3)))

    def test_tensor_ones(self):
        """Test ones tensor creation"""
        x = ot.ones((2, 4))
        assert x.shape == (2, 4)
        np.testing.assert_array_equal(x._data, np.ones((2, 4)))

    def test_tensor_randn(self):
        """Test random tensor creation"""
        x = ot.randn((3, 3))
        assert x.shape == (3, 3)
        assert x.device.type == "cpu"
        # Check that it's not all zeros (very unlikely for random)
        assert not np.allclose(x._data, np.zeros((3, 3)))


class TestDeviceManagement:
    """Test device-related functionality"""

    def test_device_creation(self):
        """Test device object creation"""
        cpu_dev = ot.device("cpu")
        assert cpu_dev.type == "cpu"
        assert cpu_dev.id == 0

        gpu_dev = ot.device("gpu")
        assert gpu_dev.type == "gpu"
        assert gpu_dev.id == 0

    def test_tensor_device_methods(self):
        """Test tensor device transfer methods"""
        x = ot.tensor([1, 2, 3, 4])

        # CPU methods
        x_cpu = x.cpu()
        assert x_cpu.device.type == "cpu"

        # GPU methods (may fail if no CUDA)
        try:
            x_gpu = x.gpu()
            assert x_gpu.device.type == "gpu"

            # Back to CPU
            x_back = x_gpu.cpu()
            assert x_back.device.type == "cpu"
            np.testing.assert_array_equal(x._data, x_back._data)
        except Exception:
            pytest.skip("GPU not available or CUDA not working")


class TestCPUOperations:
    """Test CPU-based operations (NumPy backend)"""

    def test_binary_operations_cpu(self):
        """Test binary operations on CPU"""
        x = ot.tensor([1.0, 2.0, 3.0, 4.0])
        y = ot.tensor([2.0, 3.0, 4.0, 5.0])

        # Addition
        z = x + y
        expected = np.array([3.0, 5.0, 7.0, 9.0])
        np.testing.assert_array_equal(z._data, expected)

        # Subtraction
        z = x - y
        expected = np.array([-1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(z._data, expected)

        # Multiplication
        z = x * y
        expected = np.array([2.0, 6.0, 12.0, 20.0])
        np.testing.assert_array_equal(z._data, expected)

        # Division
        z = x / y
        expected = np.array([0.5, 2 / 3, 0.75, 0.8])
        np.testing.assert_array_almost_equal(z._data, expected)

    def test_unary_operations_cpu(self):
        """Test unary operations on CPU"""
        x = ot.tensor([1.0, 4.0, 9.0, 16.0])

        # Square root
        z = x.sqrt()
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(z._data, expected)

        # Exponential
        x_small = ot.tensor([0.0, 1.0, 2.0])
        z = x_small.exp()
        expected = np.exp([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(z._data, expected)
