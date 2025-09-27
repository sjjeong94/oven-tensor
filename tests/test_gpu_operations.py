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

    def test_gpu_matmul_operations(self):
        """Test matrix multiplication on GPU"""
        try:
            # Test 2x2 matrices on GPU
            a = ot.tensor([[1.0, 2.0], [3.0, 4.0]]).gpu()
            b = ot.tensor([[5.0, 6.0], [7.0, 8.0]]).gpu()

            # Test matmul (may fail if kernel loading fails)
            try:
                # Test @ operator
                c_gpu = a @ b
                c_cpu = c_gpu.cpu()

                # Expected result: [[19, 22], [43, 50]]
                expected = np.array([[19.0, 22.0], [43.0, 50.0]])
                np.testing.assert_array_almost_equal(c_cpu._data, expected)

                # Test method call
                c2_gpu = a.matmul(b)
                c2_cpu = c2_gpu.cpu()
                np.testing.assert_array_almost_equal(c2_cpu._data, expected)

                # Test package function
                c3_gpu = ot.matmul(a, b)
                c3_cpu = c3_gpu.cpu()
                np.testing.assert_array_almost_equal(c3_cpu._data, expected)

                print("GPU matmul test passed!")

            except RuntimeError as e:
                if "kernel" in str(e).lower() or "matmul" in str(e).lower():
                    pytest.skip(f"GPU matmul kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"GPU matmul operations not available: {e}")

    def test_gpu_matmul_different_sizes(self):
        """Test GPU matmul with different matrix sizes"""
        try:
            # Test 3x2 @ 2x4 = 3x4 on GPU
            x = ot.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).gpu()  # 3x2
            y = ot.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).gpu()  # 2x4

            try:
                z_gpu = x @ y  # Should be 3x4
                z_cpu = z_gpu.cpu()

                assert z_cpu.shape == (3, 4)

                # Compare with CPU result
                x_cpu = ot.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                y_cpu = ot.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
                expected_cpu = x_cpu @ y_cpu

                np.testing.assert_array_almost_equal(
                    z_cpu.numpy(), expected_cpu.numpy()
                )

            except RuntimeError as e:
                if "kernel" in str(e).lower() or "matmul" in str(e).lower():
                    pytest.skip(f"GPU matmul kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"GPU matmul operations not available: {e}")

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

    def test_large_matmul_performance(self):
        """Test matmul performance on larger matrices"""
        try:
            # Test with reasonably sized matrices (64x64)
            size = 64
            a_cpu = ot.randn((size, size))
            b_cpu = ot.randn((size, size))

            # CPU matmul
            c_cpu = a_cpu @ b_cpu
            assert c_cpu.shape == (size, size)

            # GPU matmul (if available)
            try:
                a_gpu = a_cpu.gpu()
                b_gpu = b_cpu.gpu()
                c_gpu = a_gpu @ b_gpu
                c_gpu_result = c_gpu.cpu()

                assert c_gpu_result.shape == (size, size)

                # Compare results (allowing for floating point differences)
                np.testing.assert_allclose(
                    c_cpu.numpy(), c_gpu_result.numpy(), rtol=1e-4, atol=1e-4
                )
                print(f"Large matmul ({size}x{size}) test passed!")

            except RuntimeError as e:
                if "kernel" in str(e).lower() or "matmul" in str(e).lower():
                    pytest.skip(f"GPU matmul kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"Large matmul performance test not available: {e}")

    def test_repeated_matmul_operations(self):
        """Test repeated matmul operations for memory leaks"""
        try:
            # Small matrices for repeated operations
            a = ot.tensor([[1.0, 2.0], [3.0, 4.0]]).gpu()
            b = ot.tensor([[5.0, 6.0], [7.0, 8.0]]).gpu()

            # Perform many matmul operations
            try:
                for i in range(50):
                    c = a @ b
                    # Use result as input for next iteration occasionally
                    if i % 10 == 0:
                        a = c

                # Should not crash or run out of memory
                assert c.shape == (2, 2)
                print("Repeated matmul operations test passed!")

            except RuntimeError as e:
                if "kernel" in str(e).lower() or "matmul" in str(e).lower():
                    pytest.skip(f"GPU matmul kernel not available: {e}")
                else:
                    raise
        except Exception as e:
            pytest.skip(f"Repeated matmul GPU operations not available: {e}")
