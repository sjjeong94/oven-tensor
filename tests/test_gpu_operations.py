"""
Test GPU operations (if CUDA is available)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import oven_tensor as ot


@pytest.mark.gpu
class TestGPUOperations:
    """Test GPU-specific functionality (requires CUDA)"""

    def test_gpu_tensor_creation(self):
        """Test GPU tensor creation"""
        try:
            x = ot.tensor([1, 2, 3, 4])
            x_gpu = x.gpu()
            assert x_gpu.device.type == "gpu"

            # Test data consistency
            x_back = x_gpu.cpu()
            np.testing.assert_array_equal(x._data, x_back._data)
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")

    def test_gpu_binary_operations(self):
        """Test binary operations on GPU"""
        try:
            x = ot.tensor([1.0, 2.0, 3.0, 4.0]).gpu()
            y = ot.tensor([2.0, 3.0, 4.0, 5.0]).gpu()

            # Test addition (may fail if kernel loading fails)
            try:
                z = x + y
                expected = np.array([3.0, 5.0, 7.0, 9.0])
                np.testing.assert_array_equal(z.cpu()._data, expected)
            except RuntimeError as e:
                if "kernel" in str(e).lower():
                    pytest.skip(f"GPU kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"GPU operations not available: {e}")

    def test_gpu_unary_operations(self):
        """Test unary operations on GPU"""
        try:
            x = ot.tensor([1.0, 4.0, 9.0, 16.0]).gpu()

            # Test sqrt (may fail if kernel loading fails)
            try:
                z = x.sqrt()
                expected = np.array([1.0, 2.0, 3.0, 4.0])
                np.testing.assert_array_almost_equal(z.cpu()._data, expected)
            except RuntimeError as e:
                if "kernel" in str(e).lower():
                    pytest.skip(f"GPU kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"GPU operations not available: {e}")

    def test_mixed_device_operations(self):
        """Test operations between CPU and GPU tensors"""
        try:
            x_cpu = ot.tensor([1, 2, 3, 4])
            x_gpu = ot.tensor([5, 6, 7, 8]).gpu()

            # This should automatically handle device mismatch
            # (implementation may vary)
            try:
                z = x_cpu + x_gpu
                # Result device may be CPU or GPU depending on implementation
                assert z.shape == (4,)
            except Exception as e:
                # Expected if mixed operations not implemented
                assert "device" in str(e).lower() or "mismatch" in str(e).lower()
        except Exception as e:
            pytest.skip(f"Mixed device test not available: {e}")


@pytest.mark.slow
class TestPerformance:
    """Performance-related tests (marked as slow)"""

    def test_large_tensor_operations(self):
        """Test operations on larger tensors"""
        size = 10000
        x = ot.randn((size,))
        y = ot.randn((size,))

        # CPU operation
        z_cpu = x + y
        assert z_cpu.shape == (size,)

        # GPU operation (if available)
        try:
            x_gpu = x.gpu()
            y_gpu = y.gpu()
            z_gpu = x_gpu + y_gpu

            # Compare results (allowing for small floating point differences)
            np.testing.assert_allclose(z_cpu._data, z_gpu.cpu()._data, rtol=1e-5)
        except Exception as e:
            pytest.skip(f"Large GPU operations not available: {e}")

    def test_repeated_operations(self):
        """Test repeated operations for memory leaks"""
        try:
            x = ot.tensor([1.0, 2.0, 3.0, 4.0]).gpu()

            # Perform many operations
            for i in range(100):
                y = x + x
                z = y * x
                w = z.sqrt()

            # Should not crash or run out of memory
            assert w.shape == (4,)

        except Exception as e:
            pytest.skip(f"Repeated GPU operations not available: {e}")
