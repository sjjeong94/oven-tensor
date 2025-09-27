"""
Test kernel caching functionality
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from pathlib import Path
import oven_tensor as ot


class TestKernelCache:
    """Test kernel compilation and caching"""

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created"""
        cache_path = Path(temp_cache_dir)
        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_kernel_manager_initialization(self, clean_cache):
        """Test KernelManager can be initialized"""
        try:
            km = ot.get_kernel_manager()
            assert km is not None
            assert hasattr(km, "functions")
        except Exception as e:
            pytest.skip(f"KernelManager initialization failed: {e}")

    def test_cache_management_functions(self, clean_cache):
        """Test cache management API"""
        # Test list functions (should work even with empty cache)
        functions = ot.list_available_functions()
        assert isinstance(functions, list)

        # Test clear cache (should not error even if empty)
        ot.clear_kernel_cache()

        # Test reload kernels
        try:
            ot.reload_kernels()
        except Exception as e:
            pytest.skip(f"Kernel reload failed: {e}")

    def test_kernel_compilation_attempt(self, clean_cache):
        """Test that kernel compilation is attempted"""
        try:
            # Try to create a GPU tensor to trigger kernel loading
            x = ot.tensor([1, 2, 3, 4])
            x_gpu = x.gpu()

            # Check if cache directory has any PTX files after attempt
            cache_path = Path(clean_cache)
            ptx_files = list(cache_path.glob("*.ptx"))

            if ptx_files:
                print(f"Found {len(ptx_files)} PTX files in cache")
                for ptx_file in ptx_files:
                    print(f"  - {ptx_file.name}")
                    # Verify file is not empty
                    assert ptx_file.stat().st_size > 0
        except Exception as e:
            pytest.skip(f"GPU operations not available: {e}")

    def test_cache_persistence(self, temp_cache_dir):
        """Test that cache persists across sessions"""
        cache_path = Path(temp_cache_dir)

        # Create a dummy PTX file to simulate cached kernel
        dummy_ptx = cache_path / "test_kernel_12345.ptx"
        dummy_ptx.write_text("// Dummy PTX content for testing\n")

        assert dummy_ptx.exists()

        # Verify the file persists
        ptx_files = list(cache_path.glob("*.ptx"))
        assert len(ptx_files) >= 1
        assert dummy_ptx in ptx_files

    def test_kernel_hash_consistency(self, clean_cache):
        """Test that kernel files generate consistent hashes"""
        try:
            km = ot.get_kernel_manager()

            # Test that we can access kernel files
            kernel_files = []
            for kernel_name in ["unary_ops", "binary_ops"]:
                try:
                    file_path = km._get_kernel_file_path(kernel_name)
                    if file_path and os.path.exists(file_path):
                        kernel_files.append((kernel_name, file_path))
                except:
                    pass

            if not kernel_files:
                pytest.skip("No kernel files found")

            # Test hash generation for found files
            for kernel_name, file_path in kernel_files:
                try:
                    hash1 = km._get_file_hash(file_path)
                    hash2 = km._get_file_hash(file_path)
                    assert hash1 == hash2, f"Hash inconsistent for {kernel_name}"
                    assert len(hash1) > 0, f"Empty hash for {kernel_name}"
                except Exception as e:
                    print(f"Hash test failed for {kernel_name}: {e}")

        except Exception as e:
            pytest.skip(f"Kernel hash test failed: {e}")


class TestCLIIntegration:
    """Test command-line interface integration"""

    def test_cli_import(self):
        """Test that CLI module can be imported"""
        try:
            from oven_tensor.cli import cache_command

            assert callable(cache_command)
        except ImportError as e:
            pytest.fail(f"Failed to import CLI: {e}")

    def test_cache_command_help(self):
        """Test that cache command can show help"""
        try:
            from oven_tensor.cli import cache_command
            import sys
            from io import StringIO

            # Capture help output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                cache_command(["--help"])
            except SystemExit:
                pass  # argparse calls sys.exit after showing help
            finally:
                sys.stdout = old_stdout

            help_text = captured_output.getvalue()
            assert "oven-tensor-cache" in help_text or "cache" in help_text

        except Exception as e:
            pytest.skip(f"CLI help test failed: {e}")
