"""
Configuration and fixtures for pytest
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture(scope="session")
def temp_cache_dir():
    """Create a temporary cache directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="oven_test_")
    old_home = os.environ.get("HOME")

    # Create fake home directory structure
    fake_home = Path(temp_dir) / "fake_home"
    fake_home.mkdir()
    cache_dir = fake_home / ".oven" / "kernels"
    cache_dir.mkdir(parents=True)

    # Set HOME to temp directory for tests
    os.environ["HOME"] = str(fake_home)

    yield str(cache_dir)

    # Cleanup
    if old_home:
        os.environ["HOME"] = old_home
    else:
        del os.environ["HOME"]
    shutil.rmtree(temp_dir)


@pytest.fixture
def clean_cache(temp_cache_dir):
    """Clean cache directory before each test"""
    cache_path = Path(temp_cache_dir)
    if cache_path.exists():
        for file in cache_path.glob("*.ptx"):
            file.unlink()
    yield temp_cache_dir
