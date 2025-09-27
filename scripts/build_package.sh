#!/bin/bash

# Build and upload oven-tensor package to PyPI
set -e

echo "Building oven-tensor package..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
pip install --upgrade build twine

# Build package
python -m build

echo "Package built successfully!"
echo "Files created:"
ls -la dist/

echo ""
echo "To upload to PyPI:"
echo "1. Test PyPI: twine upload --repository testpypi dist/*"
echo "2. Real PyPI: twine upload dist/*"
echo ""
echo "To install from test PyPI:"
echo "pip install --index-url https://test.pypi.org/simple/ oven-tensor"