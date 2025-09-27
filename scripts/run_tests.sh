#!/bin/bash

# Test runner script for oven-tensor
set -e

echo "🧪 Running oven-tensor tests..."
echo "=================================="

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Run basic tests (excluding GPU and slow tests)
echo ""
echo "📋 Running basic tests..."
pytest tests/ -m "not gpu and not slow" -v

# Ask if user wants to run GPU tests
read -p "🚀 Run GPU tests? (requires CUDA) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎮 Running GPU tests..."
    pytest tests/ -m "gpu" -v || echo "⚠️  Some GPU tests failed (this is expected if CUDA is not properly configured)"
fi

# Ask if user wants to run slow tests
read -p "🐌 Run slow/performance tests? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⏱️  Running performance tests..."
    pytest tests/ -m "slow" -v
fi

echo ""
echo "✅ Test run completed!"
echo ""
echo "💡 Available test commands:"
echo "  pytest tests/                    # Run all tests"
echo "  pytest tests/ -m 'not gpu'      # Skip GPU tests"
echo "  pytest tests/ -m 'not slow'     # Skip slow tests"
echo "  pytest tests/ -v                # Verbose output"
echo "  pytest tests/ --cov=oven_tensor # With coverage"