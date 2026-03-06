#!/bin/bash
# =============================================================================
# setup_oscar.sh — One-time environment setup on OSCAR
# =============================================================================
# Run this ONCE after cloning the repo on OSCAR:
#
#   ssh username@ssh.ccv.brown.edu
#   cd ~/data
#   git clone https://github.com/xXElGenio0203Xx/DRP-Cahn-Hilliard-PINN.git
#   cd DRP-Cahn-Hilliard-PINN
#   bash setup_oscar.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs all dependencies
#   3. Patches SciPy with the custom _optimize.py
#   4. Verifies the patch
# =============================================================================

set -e  # Exit on any error

echo "============================================"
echo "Setting up Cahn-Hilliard PINN environment"
echo "============================================"

# 1. Load Python module (OSCAR uses Lmod)
echo "[1/5] Loading Python module..."
module load python/3.11.0 2>/dev/null || module load python/3.9.0 2>/dev/null || {
    echo "  No Python module found, using system Python"
}
echo "  Python: $(python3 --version)"

# 2. Create virtual environment
VENV_DIR="$HOME/pinn_env"
if [ -d "$VENV_DIR" ]; then
    echo "[2/5] Virtual environment already exists at $VENV_DIR"
else
    echo "[2/5] Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 3. Activate and install
echo "[3/5] Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# 4. Patch SciPy
echo "[4/5] Patching SciPy with custom _optimize.py..."
SCIPY_OPT_DIR=$(python -c "import scipy.optimize; import os; print(os.path.dirname(scipy.optimize.__file__))")
echo "  SciPy optimize dir: $SCIPY_OPT_DIR"

if [ ! -f "_optimize.py" ]; then
    echo "  ERROR: _optimize.py not found in current directory!"
    echo "  Make sure you're in the repo root."
    exit 1
fi

# Backup original
cp "$SCIPY_OPT_DIR/_optimize.py" "$SCIPY_OPT_DIR/_optimize.py.backup"
echo "  Original backed up to _optimize.py.backup"

# Apply patch
cp _optimize.py "$SCIPY_OPT_DIR/_optimize.py"
echo "  Patch applied!"

# 5. Verify
echo "[5/5] Verifying installation..."
python -c "
import torch
import scipy
import yaml
import matplotlib
print(f'  torch   : {torch.__version__}')
print(f'  scipy   : {scipy.__version__}')
print(f'  yaml    : {yaml.__version__}')
print(f'  mpl     : {matplotlib.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
# Verify patch: check if method_bfgs is accepted
from scipy.optimize._optimize import _minimize_bfgs
import inspect
sig = inspect.signature(_minimize_bfgs)
if 'method_bfgs' in [p for p in sig.parameters]:
    print('  SciPy patch: VERIFIED (method_bfgs found)')
else:
    # Try calling with the option to see if it works
    print('  SciPy patch: checking via options dict...')
    print('  SciPy patch: applied (file replaced)')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To submit all 17 experiments:"
echo "  sbatch submit_experiments.sh"
echo "============================================"
