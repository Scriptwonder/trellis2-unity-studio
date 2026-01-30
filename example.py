#!/usr/bin/env python3
"""
Simple example: Generate a 3D model from an image using TRELLIS.2 Unity Studio
"""

import sys
from pathlib import Path

# Add vendor to path
VENDOR_PATH = Path(__file__).parent / "vendor" / "TRELLIS.2"
sys.path.insert(0, str(VENDOR_PATH))

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

def main():
    print("="*60)
    print("TRELLIS.2 Unity Studio - Simple Example")
    print("="*60)
    
    # Load pipeline
    print("\n1. Loading TRELLIS.2 model...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    print("   ✓ Model loaded")
    
    # Load image (replace with your image path)
    image_path = "assets/examples/example1.png"
    if not Path(image_path).exists():
        print(f"\n✗ Image not found: {image_path}")
        print("  Please provide an image file or update the path in example.py")
        return
    
    print(f"\n2. Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    print(f"   ✓ Image loaded ({image.size[0]}x{image.size[1]})")
    
    # Generate 3D model
    print("\n3. Generating 3D model...")
    print("   Resolution: 1024³")
    print("   This may take ~17 seconds on H100...")
    
    result = pipeline(
        image,
        seed=42,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
    )
    print("   ✓ Generation complete!")
    
    # Export GLB
    output_path = "outputs/example_output.glb"
    Path(output_path).parent.mkdir(exist_ok=True)
    
    print(f"\n4. Exporting to GLB: {output_path}")
    glb = pipeline.postprocess(
        result['gaussian'],
        result['slat'],
        mesh_simplify=True,
        simplify_target_faces=100000,
        texture_size=1024,
    )
    glb.export(output_path)
    print(f"   ✓ Exported successfully!")
    
    # Summary
    print("\n" + "="*60)
    print("✓ Done! Your 3D model is ready:")
    print(f"  {output_path}")
    print("\nNext steps:")
    print("  1. Import the GLB into Unity (drag & drop)")
    print("  2. Or view in any GLB viewer")
    print("  3. Try the full app: python app.py")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Ensure vendor/TRELLIS.2 is set up: bash vendor/TRELLIS.2/setup.sh")
        print("  - Check GPU availability: python -c 'import torch; print(torch.cuda.is_available())'")
        print("  - See docs/troubleshooting.md for more help")
