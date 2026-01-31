"""
TRELLIS.2 + Flux Wrapper
Provides inference functions for text-to-3D and image-to-3D pipelines.

Memory modes (set via MEMORY_MODE env var):
  - "keep_loaded": Both models stay in memory (requires 64GB+ RAM, fastest)
  - "swap":        Unload Flux before loading TRELLIS.2 (works with 32GB RAM)
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Set attention backends to flash_attn if you have GPU support
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_CONV_BACKEND', 'flex_gemm')

import gc
import time
import psutil
from dataclasses import dataclass
from typing import Optional, Literal, Callable

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# Minimum RAM (in GB) to safely keep both models loaded
MIN_RAM_KEEP_LOADED = 48

# Quality presets
QUALITY_PRESETS = {
    'superfast': {
        'pipeline_type': '512',
        'inference_steps': 4,
        'decimation_target': 30000,
        'texture_size': 512,
        'remesh': False,
        # Optimized sampler params for maximum speed
        'ss_guidance_strength': 5.0,
        'shape_guidance_strength': 5.0,
        'tex_guidance_strength': 1.0,
        'use_compile': True,  # torch.compile for speed
    },
    'fast': {
        'pipeline_type': '512',
        'inference_steps': 15,
        'decimation_target': 50000,
        'texture_size': 1024,
        'remesh': False,
    },
    'balanced': {
        'pipeline_type': '512',
        'inference_steps': 25,
        'decimation_target': 100000,
        'texture_size': 2048,
        'remesh': False,
    },
    'high': {
        'pipeline_type': '1024_cascade',
        'inference_steps': 50,
        'decimation_target': 500000,
        'texture_size': 4096,
        'remesh': True,
    },
}

QualityLevel = Literal['superfast', 'fast', 'balanced', 'high']

# Progress stages reported during generation
STAGES = {
    'loading_flux': 'Loading Flux model',
    'generating_image': 'Generating image',
    'unloading_flux': 'Freeing GPU memory',
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


def _detect_memory_mode() -> str:
    """Auto-detect memory mode based on available system RAM."""
    mode = os.environ.get('MEMORY_MODE', 'auto')
    if mode in ('keep_loaded', 'swap'):
        return mode

    # Auto-detect based on available RAM
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_ram_gb >= MIN_RAM_KEEP_LOADED:
        print(f"[INFO] Detected {total_ram_gb:.0f}GB RAM -> keep_loaded mode")
        return 'keep_loaded'
    else:
        print(f"[INFO] Detected {total_ram_gb:.0f}GB RAM (< {MIN_RAM_KEEP_LOADED}GB) -> swap mode")
        return 'swap'


class Trellis2Pipeline:
    """
    Combined Flux + TRELLIS.2 pipeline for text-to-3D and image-to-3D.

    Memory modes:
        keep_loaded: Both models stay in memory. Fastest for repeated text-to-3D
                     calls since no model loading/unloading between requests.
                     Requires 64GB+ system RAM.

        swap:        Flux is unloaded before TRELLIS.2 runs. Works with 32GB RAM
                     but adds ~50s overhead per text-to-3D call for model swapping.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        flux_model: str = "black-forest-labs/FLUX.2-klein-4B",
        trellis_model: str = "microsoft/TRELLIS.2-4B",
        preload_flux: bool = False,
        preload_trellis: bool = True,
        memory_mode: str = "auto",
    ):
        self.device = device
        self.dtype = dtype
        self.flux_model = flux_model
        self.trellis_model = trellis_model

        self._flux_pipe: Optional[Flux2KleinPipeline] = None
        self._trellis_pipe: Optional[Trellis2ImageTo3DPipeline] = None

        # Determine memory strategy
        if memory_mode == "auto":
            self.memory_mode = _detect_memory_mode()
        else:
            self.memory_mode = memory_mode

        print(f"[INFO] Memory mode: {self.memory_mode}")

        if self.memory_mode == 'keep_loaded':
            # In keep_loaded mode, preload both models
            if preload_flux:
                self._load_flux()
            if preload_trellis:
                self._load_trellis()
        else:
            # In swap mode, don't preload - models are loaded on demand
            # to avoid having both in memory simultaneously
            print("[INFO] Swap mode: models will be loaded on demand.")

    def _load_flux(self) -> Flux2KleinPipeline:
        """Load Flux pipeline (lazy loading)."""
        if self._flux_pipe is None:
            print("[INFO] Loading Flux pipeline...")
            self._flux_pipe = Flux2KleinPipeline.from_pretrained(
                self.flux_model,
                torch_dtype=self.dtype
            )
            self._flux_pipe.enable_model_cpu_offload()
            print("[INFO] Flux pipeline loaded.")
        return self._flux_pipe

    def _load_trellis(self, use_compile: bool = False) -> Trellis2ImageTo3DPipeline:
        """Load TRELLIS.2 pipeline (lazy loading)."""
        if self._trellis_pipe is None:
            print("[INFO] Loading TRELLIS.2 pipeline...")
            
            # Clear any leftover torch function mode stack from Flux cpu_offload
            # This prevents meta tensor issues in BiRefNet model loading
            try:
                import torch.utils._device as _device_module
                # Pop all modes from the torch function stack
                while _device_module._len_torch_function_stack() > 0:
                    _device_module._pop_mode()
                    print("[DEBUG] Popped a torch function mode")
            except Exception as e:
                print(f"[DEBUG] Could not clear torch function stack: {e}")
            
            self._trellis_pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                self.trellis_model
            )
            self._trellis_pipe.cuda()
            self._trellis_pipe.low_vram = True
            print("[INFO] TRELLIS.2 pipeline loaded.")
        
        # Apply torch.compile for superfast mode (cached after first run)
        if use_compile and not getattr(self, '_compiled', False):
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

    def _unload_flux(self):
        """Unload Flux to free memory."""
        if self._flux_pipe is not None:
            # Remove cpu_offload hooks before deleting to clean up accelerate state
            try:
                self._flux_pipe.maybe_free_model_hooks()
            except Exception:
                pass
            del self._flux_pipe
            self._flux_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reset torch device context stack to prevent meta tensor issues
            # This is needed because enable_model_cpu_offload() may leave
            # device contexts that cause torch.linspace() to create meta tensors
            try:
                from torch.utils._device import _device_constructors, _caching_mode
                # Clear any registered device constructors that might redirect to meta device
                if hasattr(torch.utils._device, '_device_constructors'):
                    torch.utils._device._device_constructors.clear()
            except Exception:
                pass
            
            print("[INFO] Flux pipeline unloaded.")

    def _unload_trellis(self):
        """Unload TRELLIS.2 to free memory."""
        if self._trellis_pipe is not None:
            del self._trellis_pipe
            self._trellis_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[INFO] TRELLIS.2 pipeline unloaded.")

    def generate_image(
        self,
        prompt: str,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
    ) -> Image.Image:
        """Generate an image from a text prompt using Flux."""
        flux = self._load_flux()

        image = flux(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(seed)
        ).images[0]

        return image

    def image_to_3d(
        self,
        image: Image.Image,
        output_dir: str,
        output_name: str = "output",
        quality: QualityLevel = "balanced",
        seed: int = 42,
        on_progress: ProgressCallback = None,
    ) -> InferenceResult:
        """Generate a 3D model from an image using TRELLIS.2."""
        os.makedirs(output_dir, exist_ok=True)
        preset = QUALITY_PRESETS[quality]
        timings = {}

        def _report(stage: str):
            if on_progress:
                on_progress(stage, STAGES.get(stage, stage))

        # Load TRELLIS.2
        _report('loading_trellis')
        t0 = time.time()
        use_compile = preset.get('use_compile', False)
        trellis = self._load_trellis(use_compile=use_compile)
        timings['trellis_load'] = time.time() - t0

        # Build sampler params (superfast uses optimized guidance values)
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
        on_progress: ProgressCallback = None,
    ) -> InferenceResult:
        """
        Generate a 3D model from a text prompt (Flux + TRELLIS.2).

        In 'keep_loaded' mode, both models stay in memory for fastest throughput.
        In 'swap' mode, Flux is unloaded before TRELLIS.2 runs (saves ~20GB RAM).

        Args:
            prompt: Text description of the object to generate
            output_dir: Directory to save output files
            output_name: Prefix for output filenames
            quality: Quality preset ('fast', 'balanced', 'high')
            seed: Random seed for reproducibility
            on_progress: Optional callback for progress reporting

        Returns:
            InferenceResult with paths to generated files and timing info
        """
        os.makedirs(output_dir, exist_ok=True)
        timings = {}

        def _report(stage: str):
            if on_progress:
                on_progress(stage, STAGES.get(stage, stage))

        # In swap mode, unload TRELLIS.2 before loading Flux
        if self.memory_mode == 'swap':
            self._unload_trellis()

        # Generate image with Flux
        _report('loading_flux')
        t0 = time.time()
        _report('generating_image')
        image = self.generate_image(prompt, seed=seed)
        timings['flux_generate'] = time.time() - t0

        # Save image
        image_path = os.path.join(output_dir, f"{output_name}_image.png")
        image.save(image_path)

        # In swap mode, free Flux memory before loading TRELLIS.2
        if self.memory_mode == 'swap':
            _report('unloading_flux')
            self._unload_flux()

        # Generate 3D from image
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
        _pipeline = Trellis2Pipeline(
            preload_trellis=True,
            memory_mode="auto",
        )
    return _pipeline


def run_text_to_3d(
    prompt: str,
    output_dir: str,
    output_name: str = "output",
    quality: QualityLevel = "balanced",
    seed: int = 42,
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
print("[INFO] Initializing pipeline...")
_pipeline = Trellis2Pipeline(preload_trellis=True, memory_mode="auto")
print("[INFO] Pipeline ready.")
