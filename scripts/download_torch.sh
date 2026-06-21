#!/bin/bash
# Download PyTorch wheel ONCE for Docker build
# Run: bash scripts/download_torch.sh

echo "Downloading PyTorch wheel (532MB, one-time download)..."
echo "This file will be cached for all future Docker builds."
echo ""

# Download with resume support (-c)
wget -c --progress=bar \
  "https://files.pythonhosted.org/packages/3c/0d/torch-2.12.1-cp312-cp312-manylinux_2_28_x86_64.whl" \
  -O torch-2.12.1-cp312-cp312-manylinux_2_28_x86_64.whl

echo ""
echo "Done! PyTorch wheel downloaded. Now build Docker:"
echo "  docker build -t neurocuda:ros2 -f Dockerfile.ros2 ."
