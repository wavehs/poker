#!/usr/bin/env bash
# Dev environment setup script
# Usage: bash infra/dev/setup.sh

set -euo pipefail

echo "♠️ Poker Helper — Dev Setup"
echo "=========================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate 2>/dev/null || .venv/Scripts/activate 2>/dev/null || true

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-dev.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw_frames
mkdir -p data/labeled_frames
mkdir -p data/synthetic_tables/output
mkdir -p data/sample_frames

# Generate initial synthetic data
echo "Generating synthetic frames..."
python -m data.synthetic_tables.generator

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  make api    — Start API server on :8000"
echo "  make ui     — Start UI server on :3000"
echo "  make test   — Run all tests"
echo "  make synth  — Generate synthetic frames"
