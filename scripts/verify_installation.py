#!/usr/bin/env python3
"""
Verification script for TRELLIS.2 Unity Studio installation.
Checks all dependencies and system requirements.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_check(name, status, details=""):
    symbol = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    print(f"{symbol} {name:<30} {details}")

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 8)
    status = version >= required
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    print_check("Python Version", status, details)
    return status

def check_module(module_name, display_name=None):
    """Check if a Python module is installed."""
    if display_name is None:
        display_name = module_name
    
    try:
        importlib.import_module(module_name)
        print_check(display_name, True, "installed")
        return True
    except ImportError:
        print_check(display_name, False, "NOT FOUND")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            details = f"{device_name} ({memory_gb:.1f}GB, CUDA {cuda_version})"
            print_check("CUDA Support", True, details)
            
            # Warn if low memory
            if memory_gb < 24:
                print(f"  {Colors.YELLOW}⚠ Warning: Less than 24GB VRAM. Use lower resolutions.{Colors.END}")
            
            return True
        else:
            print_check("CUDA Support", False, "No CUDA devices found")
            return False
    except Exception as e:
        print_check("CUDA Support", False, f"Error: {e}")
        return False

def check_vendor_setup():
    """Check if vendor/TRELLIS.2 is set up."""
    vendor_path = Path(__file__).parent.parent / "vendor" / "TRELLIS.2"
    
    # Check if submodule exists
    if not vendor_path.exists():
        print_check("TRELLIS.2 Submodule", False, "Not initialized")
        print(f"  {Colors.YELLOW}Run: git submodule update --init --recursive{Colors.END}")
        return False
    
    # Check if Python package is importable
    sys.path.insert(0, str(vendor_path))
    try:
        import trellis2
        print_check("TRELLIS.2 Package", True, "installed")
        
        # Try importing key components
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        print_check("TRELLIS.2 Pipeline", True, "available")
        return True
    except ImportError as e:
        print_check("TRELLIS.2 Package", False, "Not built")
        print(f"  {Colors.YELLOW}Run: cd vendor/TRELLIS.2 && bash setup.sh --basic --o-voxel{Colors.END}")
        return False

def check_directories():
    """Check if required directories exist."""
    project_root = Path(__file__).parent.parent
    required_dirs = {
        "outputs": "Output directory",
        "tmp": "Temporary files",
        "assets": "Assets directory",
    }
    
    all_exist = True
    for dir_name, description in required_dirs.items():
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print_check(description, exists, str(dir_path))
        if not exists:
            all_exist = False
    
    return all_exist

def check_model_access():
    """Check if models can be accessed from Hugging Face."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_id = "microsoft/TRELLIS.2-4B"
        
        # Check if model exists and is accessible
        try:
            model_info = api.model_info(model_id)
            print_check("Model Access", True, f"{model_id}")
            return True
        except Exception:
            print_check("Model Access", False, "May need authentication")
            print(f"  {Colors.YELLOW}Models will download on first use{Colors.END}")
            return True  # Not critical
    except ImportError:
        print_check("Model Access", False, "huggingface_hub not installed")
        return False

def main():
    print_header("TRELLIS.2 Unity Studio - Installation Verification")
    
    results = {
        'critical': [],
        'optional': [],
    }
    
    # Critical checks
    print(f"\n{Colors.BOLD}Critical Components:{Colors.END}")
    results['critical'].append(check_python_version())
    results['critical'].append(check_module('torch', 'PyTorch'))
    results['critical'].append(check_module('gradio', 'Gradio'))
    results['critical'].append(check_module('numpy', 'NumPy'))
    results['critical'].append(check_module('PIL', 'Pillow'))
    results['critical'].append(check_cuda())
    results['critical'].append(check_vendor_setup())
    
    # Optional checks
    print(f"\n{Colors.BOLD}Optional Components:{Colors.END}")
    results['optional'].append(check_module('cv2', 'OpenCV'))
    results['optional'].append(check_module('trimesh', 'Trimesh'))
    results['optional'].append(check_directories())
    results['optional'].append(check_model_access())
    
    # Summary
    print_header("Verification Summary")
    
    critical_passed = sum(results['critical'])
    critical_total = len(results['critical'])
    optional_passed = sum(results['optional'])
    optional_total = len(results['optional'])
    
    print(f"Critical: {critical_passed}/{critical_total} passed")
    print(f"Optional: {optional_passed}/{optional_total} passed")
    
    if critical_passed == critical_total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Installation verified successfully!{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print(f"  1. Launch the app: python app.py")
        print(f"  2. Open browser to: http://localhost:7860")
        print(f"  3. See QUICKSTART.md for usage guide")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Installation incomplete{Colors.END}")
        print(f"\n{Colors.BOLD}Please fix the issues above and run verification again.{Colors.END}")
        print(f"See docs/installation.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
