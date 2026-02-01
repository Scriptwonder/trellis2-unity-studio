"""
TRELLIS.2 + GPT-Image API Wrapper
Provides inference functions for text-to-3D and image-to-3D pipelines.

This version uses OpenAI's GPT-Image API for text-to-image generation,
offloading image generation to the cloud. This allows TRELLIS.2 to stay
loaded in GPU memory at all times, eliminating model swapping overhead.

Benefits:
  - No model swapping: TRELLIS.2 stays loaded 100% of the time
  - No local GPU memory for image generation
  - Simpler memory management
  - Faster text-to-3D (no ~50s swap overhead)

Requirements:
  - OPENAI_API_KEY environment variable must be set
  - openai package installed (pip install openai)
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Set attention backends to flash_attn if you have GPU support
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_CONV_BACKEND', 'flex_gemm')

import io
import base64
import time
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import torch
from PIL import Image
from openai import OpenAI
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# Quality presets for TRELLIS.2
QUALITY_PRESETS = {
    'superfast': {
        'pipeline_type': '512',
        'inference_steps': 4,
        'decimation_target': 20000,
        'texture_size': 512,
        'remesh': False,
        # Optimized sampler params for maximum speed
        'ss_guidance_strength': 5.0,
        'shape_guidance_strength': 5.0,
        'tex_guidance_strength': 1.0,
        'low_vram': True,  # Disable low_vram for speed (requires 24GB+)
    },
    'fast': {
        'pipeline_type': '512',
        'inference_steps': 12,
        'decimation_target': 50000,
        'texture_size': 1024,
        'remesh': False,
        'low_vram': True,
    },
    'balanced': {
        'pipeline_type': '512',
        'inference_steps': 25,
        'decimation_target': 100000,
        'texture_size': 2048,
        'remesh': False,
        'low_vram': True,
    },
    'high': {
        'pipeline_type': '1024_cascade',
        'inference_steps': 50,
        'decimation_target': 500000,
        'texture_size': 4096,
        'remesh': True,
        'low_vram': True,
    },
}

QualityLevel = Literal['superfast', 'fast', 'balanced', 'high']

# Progress stages reported during generation
STAGES = {
    'generating_image': 'Generating image (GPT-Image API)',
    'loading_trellis': 'Loading TRELLIS.2 model',
    'generating_mesh': 'Generating 3D mesh',
    'exporting_glb': 'Exporting GLB',
}

# Callable type for progress reporting: (stage_key, stage_description) -> None
ProgressCallback = Optional[Callable[[str, str], None]]


@dataclass
class InferenceResult:
    """Result of an inference run."""
    glb_path: str
    image_path: Optional[str] = None
    timings: Optional[dict] = None


class GPTImageClient:
    """
    Client for OpenAI's GPT-Image API.
    
    Handles text-to-image generation via the cloud API,
    freeing local GPU for TRELLIS.2.
    """
    
    def __init__(
        self,
        model: str = "gpt-image-1.5",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the GPT-Image client.
        
        Args:
            model: GPT-Image model to use (default: gpt-image-1.5)
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        print(f"[INFO] GPT-Image client initialized (model: {model})")
    
    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (e.g., "1024x1024", "1792x1024")
            quality: Image quality ("standard" or "hd")
            
        Returns:
            PIL Image object
        """
        print(f"[INFO] Generating image via GPT-Image API...")
        print(f"[INFO] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # GPT-Image API returns b64_json by default
        result = self._client.images.generate(
            model=self.model,
            prompt=prompt,
        )
        
        # Decode base64 response to PIL Image
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (remove alpha channel for TRELLIS.2)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        print(f"[INFO] Image generated: {image.size[0]}x{image.size[1]}")
        return image


class Trellis2Pipeline:
    """
    TRELLIS.2 pipeline with GPT-Image API for text-to-image.
    
    This pipeline keeps TRELLIS.2 loaded at all times since image
    generation is offloaded to the cloud via GPT-Image API.
    No model swapping required!
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trellis_model: str = "microsoft/TRELLIS.2-4B",
        gpt_image_model: str = "gpt-image-1.5",
        openai_api_key: Optional[str] = None,
        preload_trellis: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            device: Device for TRELLIS.2 (default: cuda)
            dtype: Data type for TRELLIS.2 (default: bfloat16)
            trellis_model: TRELLIS.2 model ID on HuggingFace
            gpt_image_model: GPT-Image model to use
            openai_api_key: OpenAI API key (uses env var if None)
            preload_trellis: Whether to load TRELLIS.2 immediately
        """
        self.device = device
        self.dtype = dtype
        self.trellis_model = trellis_model

        # Initialize GPT-Image client (no GPU memory needed!)
        self._gpt_image = GPTImageClient(
            model=gpt_image_model,
            api_key=openai_api_key,
        )

        # TRELLIS.2 pipeline (stays loaded)
        self._trellis_pipe: Optional[Trellis2ImageTo3DPipeline] = None
        self._compiled = False

        if preload_trellis:
            self._load_trellis()

    def _load_trellis(self, use_compile: bool = False, low_vram: bool = True) -> Trellis2ImageTo3DPipeline:
        """Load TRELLIS.2 pipeline (lazy loading)."""
        if self._trellis_pipe is None:
            print("[INFO] Loading TRELLIS.2 pipeline...")
            
            self._trellis_pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                self.trellis_model
            )
            self._trellis_pipe.cuda()
            print("[INFO] TRELLIS.2 pipeline loaded and ready.")
        
        # Configure low_vram mode (can be changed per-request)
        self._trellis_pipe.low_vram = low_vram
        if not low_vram:
            print("[INFO] Low-VRAM mode disabled for maximum speed (requires 24GB+ VRAM)")
        
        # Apply torch.compile for superfast mode (cached after first run)
        if use_compile and not self._compiled:
            try:
                print("[INFO] Compiling models with torch.compile (first run will be slow)...")
                if 'sparse_structure_flow_model' in self._trellis_pipe.models:
                    self._trellis_pipe.models['sparse_structure_flow_model'] = torch.compile(
                        self._trellis_pipe.models['sparse_structure_flow_model'],
                        mode='reduce-overhead'
                    )
                if 'shape_slat_flow_model_512' in self._trellis_pipe.models:
                    self._trellis_pipe.models['shape_slat_flow_model_512'] = torch.compile(
                        self._trellis_pipe.models['shape_slat_flow_model_512'],
                        mode='reduce-overhead'
                    )
                self._compiled = True
                print("[INFO] Models compiled successfully.")
            except Exception as e:
                print(f"[WARN] torch.compile failed (continuing without): {e}")
        
        return self._trellis_pipe

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Image.Image:
        """
        Generate an image from a text prompt using GPT-Image API.
        
        This runs in the cloud, not on local GPU!
        
        Args:
            prompt: Text description of the image
            size: Image size (default: 1024x1024)
            quality: Image quality ("standard" or "hd")
            
        Returns:
            PIL Image object
        """
        return self._gpt_image.generate(
            prompt=prompt,
            size=size,
            quality=quality,
        )

    def image_to_3d(
        self,
        image: Image.Image,
        output_dir: str,
        output_name: str = "output",
        quality: QualityLevel = "balanced",
        seed: int = 42,
        on_progress: ProgressCallback = None,
    ) -> InferenceResult:
        """
        Generate a 3D model from an image using TRELLIS.2.
        
        TRELLIS.2 is always loaded, so this is very fast!
        """
        os.makedirs(output_dir, exist_ok=True)
        preset = QUALITY_PRESETS[quality]
        timings = {}

        def _report(stage: str):
            if on_progress:
                on_progress(stage, STAGES.get(stage, stage))

        # Load TRELLIS.2 (should already be loaded, this is a no-op)
        _report('loading_trellis')
        t0 = time.time()
        use_compile = preset.get('use_compile', False)
        low_vram = preset.get('low_vram', True)
        trellis = self._load_trellis(use_compile=use_compile, low_vram=low_vram)
        timings['trellis_load'] = time.time() - t0

        # Build sampler params
        ss_params = {
            'steps': preset['inference_steps'],
        }
        shape_params = {
            'steps': preset['inference_steps'],
        }
        tex_params = {
            'steps': preset['inference_steps'],
        }
        
        # Apply optimized guidance for superfast mode
        if 'ss_guidance_strength' in preset:
            ss_params['guidance_strength'] = preset['ss_guidance_strength']
        if 'shape_guidance_strength' in preset:
            shape_params['guidance_strength'] = preset['shape_guidance_strength']
        if 'tex_guidance_strength' in preset:
            tex_params['guidance_strength'] = preset['tex_guidance_strength']

        # Generate 3D mesh
        _report('generating_mesh')
        t0 = time.time()
        mesh = trellis.run(
            image,
            seed=seed,
            pipeline_type=preset['pipeline_type'],
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params
        )[0]
        timings['trellis_generate'] = time.time() - t0

        # Export GLB
        _report('exporting_glb')
        t0 = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=preset['decimation_target'],
            texture_size=preset['texture_size'],
            remesh=preset['remesh'],
            remesh_band=1,
            remesh_project=0,
            verbose=False
        )

        glb_path = os.path.join(output_dir, f"{output_name}.glb")
        glb.export(glb_path, extension_webp=False)
        timings['export_glb'] = time.time() - t0

        return InferenceResult(glb_path=glb_path, timings=timings)

    def text_to_3d(
        self,
        prompt: str,
        output_dir: str,
        output_name: str = "output",
        quality: QualityLevel = "balanced",
        seed: int = 42,
        image_size: str = "1024x1024",
        image_quality: str = "standard",
        on_progress: ProgressCallback = None,
    ) -> InferenceResult:
        """
        Generate a 3D model from a text prompt.
        
        Uses GPT-Image API for text-to-image (cloud), then TRELLIS.2 for
        image-to-3D (local GPU). No model swapping required!

        Args:
            prompt: Text description of the object to generate
            output_dir: Directory to save output files
            output_name: Prefix for output filenames
            quality: TRELLIS.2 quality preset ('superfast', 'fast', 'balanced', 'high')
            seed: Random seed for TRELLIS.2 reproducibility
            image_size: GPT-Image output size (default: 1024x1024)
            image_quality: GPT-Image quality ("standard" or "hd")
            on_progress: Optional callback for progress reporting

        Returns:
            InferenceResult with paths to generated files and timing info
        """
        os.makedirs(output_dir, exist_ok=True)
        timings = {}

        def _report(stage: str):
            if on_progress:
                on_progress(stage, STAGES.get(stage, stage))

        # Generate image with GPT-Image API (cloud - no local GPU!)
        _report('generating_image')
        t0 = time.time()
        image = self.generate_image(
            prompt=prompt,
            size=image_size,
            quality=image_quality,
        )
        timings['gpt_image_generate'] = time.time() - t0

        # Save image
        image_path = os.path.join(output_dir, f"{output_name}_image.png")
        image.save(image_path)
        print(f"[INFO] Saved generated image to {image_path}")

        # Generate 3D from image (TRELLIS.2 is already loaded!)
        result = self.image_to_3d(
            image=image,
            output_dir=output_dir,
            output_name=output_name,
            quality=quality,
            seed=seed,
            on_progress=on_progress,
        )

        # Merge timings
        result.timings = {**timings, **result.timings}
        result.image_path = image_path

        return result


# Global pipeline instance (loaded on import)
_pipeline: Optional[Trellis2Pipeline] = None


def get_pipeline() -> Trellis2Pipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Trellis2Pipeline(preload_trellis=True)
    return _pipeline


def run_text_to_3d(
    prompt: str,
    output_dir: str,
    output_name: str = "output",
    quality: QualityLevel = "balanced",
    seed: int = 42,
    image_size: str = "1024x1024",
    image_quality: str = "standard",
    on_progress: ProgressCallback = None,
) -> InferenceResult:
    """Convenience function for text-to-3D generation."""
    pipeline = get_pipeline()
    return pipeline.text_to_3d(
        prompt=prompt,
        output_dir=output_dir,
        output_name=output_name,
        quality=quality,
        seed=seed,
        image_size=image_size,
        image_quality=image_quality,
        on_progress=on_progress,
    )


def run_image_to_3d(
    image: Image.Image,
    output_dir: str,
    output_name: str = "output",
    quality: QualityLevel = "balanced",
    seed: int = 42,
    on_progress: ProgressCallback = None,
) -> InferenceResult:
    """Convenience function for image-to-3D generation."""
    pipeline = get_pipeline()
    return pipeline.image_to_3d(
        image=image,
        output_dir=output_dir,
        output_name=output_name,
        quality=quality,
        seed=seed,
        on_progress=on_progress,
    )


# Preload on module import
print("[INFO] Initializing TRELLIS.2 pipeline (GPT-Image mode)...")
print("[INFO] Image generation will use OpenAI GPT-Image API (cloud)")
print("[INFO] TRELLIS.2 will stay loaded - no model swapping needed!")
_pipeline = Trellis2Pipeline(preload_trellis=True)
print("[INFO] Pipeline ready.")
