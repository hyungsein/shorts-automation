"""
🎨 Image Agent - Generates images using diffusers (local, no server needed)
"""

import asyncio
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image

from ..models import ImageResult
from .base import BaseAgent


class ImageAgent(BaseAgent[list[ImageResult]]):
    """Agent for generating images with Stable Diffusion XL (diffusers)"""

    # Model path - Animagine XL 3.1
    MODEL_PATH = Path.home(
    ) / "ComfyUI" / "models" / "checkpoints" / "animagine-xl-3.1.safetensors"

    # HuggingFace fallback model (if local not found)
    HF_MODEL = "cagliostrolab/animagine-xl-3.1"

    @property
    def name(self) -> str:
        return "🎨 ImageAgent"

    # 쇼츠용 카툰 스타일 기본 프롬프트
    CHARACTER_BASE_PROMPT = """
    masterpiece, best quality, anime style illustration,
    korean webtoon style, vibrant colors, clean lines,
    detailed background, cinematic lighting,
    emotional expression, dynamic pose
    """.strip()

    NEGATIVE_PROMPT = """
    ugly, deformed, noisy, blurry, low quality, 
    bad anatomy, bad proportions, watermark, text,
    realistic, photo, 3d render, nsfw, nude,
    extra fingers, mutated hands, poorly drawn face
    """.strip()

    def __init__(self):
        super().__init__()
        self._pipe: Optional[StableDiffusionXLPipeline] = None

    def _load_pipeline(self) -> StableDiffusionXLPipeline:
        """Load the Stable Diffusion XL pipeline (lazy loading)"""
        if self._pipe is not None:
            return self._pipe

        self.log(
            "Loading Stable Diffusion XL model (first time may take a while)..."
        )

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            device = "mps"
            # MPS에서는 float32가 더 안정적 (float16은 검은 화면 문제 발생)
            dtype = torch.float32
            self.log("Using Apple Silicon MPS acceleration 🍎")
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            self.log("Using NVIDIA CUDA acceleration 🟢")
        else:
            device = "cpu"
            dtype = torch.float32
            self.log("Using CPU (this will be slow) 🐢")

        # Try local model first, then HuggingFace
        if self.MODEL_PATH.exists():
            self.log(f"Loading local model: {self.MODEL_PATH.name}")
            self._pipe = StableDiffusionXLPipeline.from_single_file(
                str(self.MODEL_PATH),
                torch_dtype=dtype,
                use_safetensors=True,
            )
        else:
            self.log(
                f"Local model not found, downloading from HuggingFace: {self.HF_MODEL}"
            )
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self.HF_MODEL,
                torch_dtype=dtype,
                use_safetensors=True,
            )

        # Use faster scheduler
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config)

        self._pipe = self._pipe.to(device)

        # Memory optimization
        self._pipe.enable_attention_slicing()

        self.log("Model loaded successfully! ✨")
        return self._pipe

    async def run(
            self,
            prompts: list[str],
            output_dir: Path,
            character_prompt: Optional[str] = None,
            width: int = 1024,  # SDXL works best with 1024
            height: int = 1024,  # Will resize for shorts later
    ) -> list[ImageResult]:
        """Generate multiple images for the video"""
        self.log(f"Generating {len(prompts)} images...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_prompt = character_prompt or self.CHARACTER_BASE_PROMPT
        results = []

        # Load pipeline once
        pipe = self._load_pipeline()

        for i, scene_prompt in enumerate(prompts):
            # Combine character base + scene-specific prompt
            full_prompt = f"{base_prompt}, {scene_prompt}"

            self.log(f"Generating image {i+1}/{len(prompts)}...")

            image_path = output_dir / f"image_{i:03d}.png"

            try:
                # Run generation in thread pool (sync -> async)
                image = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._generate_sync(pipe, full_prompt, width,
                                                      height))

                # Resize to shorts format (9:16) - 1080x1920
                shorts_image = self._resize_for_shorts(image)
                shorts_image.save(image_path)

                results.append(
                    ImageResult(
                        file_path=image_path,
                        prompt=full_prompt,
                        index=i,
                    ))
                self.log(f"✓ Image {i+1} saved")
            except Exception as e:
                self.log(f"Failed to generate image {i}: {e}")

        self.log(f"Generated {len(results)} images")
        return results

    def _generate_sync(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str,
        width: int,
        height: int,
    ) -> Image.Image:
        """Synchronous image generation (called in thread pool)"""
        result = pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=25,
            guidance_scale=7.0,
        )
        return result.images[0]

    def _resize_for_shorts(self, image: Image.Image) -> Image.Image:
        """Resize and crop image to 9:16 shorts format (1080x1920)"""
        target_w, target_h = 1080, 1920
        target_ratio = target_h / target_w  # 1.777...

        img_w, img_h = image.size
        img_ratio = img_h / img_w

        if img_ratio < target_ratio:
            # Image is wider - crop sides
            new_w = int(img_h / target_ratio)
            left = (img_w - new_w) // 2
            image = image.crop((left, 0, left + new_w, img_h))
        else:
            # Image is taller - crop top/bottom
            new_h = int(img_w * target_ratio)
            top = (img_h - new_h) // 2
            image = image.crop((0, top, img_w, top + new_h))

        # Resize to target
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    async def generate_character_sheet(
        self,
        character_description: str,
        output_dir: Path,
        num_variations: int = 5,
    ) -> list[ImageResult]:
        """Generate multiple variations of the same character"""

        expressions = [
            "happy expression, smiling",
            "surprised expression, shocked",
            "thinking expression, curious",
            "sad expression, melancholy",
            "excited expression, energetic",
        ]

        prompts = [
            f"{character_description}, {expr}"
            for expr in expressions[:num_variations]
        ]

        return await self.run(prompts, output_dir)

    async def check_connection(self) -> bool:
        """Check if model is available (local file or can download)"""
        if self.MODEL_PATH.exists():
            return True
        # Can always download from HuggingFace
        return True
