#!/bin/bash
# TRELLIS.2 Unity Studio - Setup Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "  TRELLIS.2 Unity Studio Setup"
echo "=============================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓${NC} CUDA found: $(nvcc --version | grep release)"
else
    echo -e "${YELLOW}⚠${NC} CUDA not found - GPU support may not work"
fi

# Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# Install PyTorch
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Setup vendor TRELLIS.2
echo "Setting up TRELLIS.2..."
cd vendor/TRELLIS.2
if [ -f "setup.sh" ]; then
    bash setup.sh --basic --flash-attn --nvdiffrast --cumesh --o-voxel
fi
cd "$PROJECT_ROOT"

# Create directories
mkdir -p outputs

echo ""
echo "=============================================="
echo -e "${GREEN}✓ Setup complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Start server:  uvicorn src.trellis2_server:app --port 8000"
echo "  2. Web UI:        python app.py"
echo "  3. Unity:         Copy unity/ to Assets/Trellis2/"
echo ""
