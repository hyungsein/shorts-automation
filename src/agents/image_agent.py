"""
🎨 Image Agent - Generates images using Stable Diffusion (local)
"""

import base64
import io
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

from ..config import settings
from ..models import ImageResult
from .base import BaseAgent


class ImageAgent(BaseAgent[list[ImageResult]]):
    """Agent for generating images with Stable Diffusion"""

    @property
    def name(self) -> str:
        return "🎨 ImageAgent"

    # Default character prompt (customize as needed)
    CHARACTER_BASE_PROMPT = """
    anime style, illustration, attractive female character, 
    stylish outfit, detailed, high quality, vibrant colors,
    upper body shot, expressive face
    """.strip()

    NEGATIVE_PROMPT = """
    ugly, deformed, noisy, blurry, low quality, 
    bad anatomy, bad proportions, watermark, text,
    realistic, photo, 3d render
    """.strip()

    def __init__(self):
        super().__init__()
        self.api_url = settings.sd.api_url

    async def run(
            self,
            prompts: list[str],
            output_dir: Path,
            character_prompt: Optional[str] = None,
            width: int = 1080,
            height: int = 1920,  # 9:16 for Shorts
    ) -> list[ImageResult]:
        """Generate multiple images for the video"""
        self.log(f"Generating {len(prompts)} images...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_prompt = character_prompt or self.CHARACTER_BASE_PROMPT
        results = []

        for i, scene_prompt in enumerate(prompts):
            # Combine character base + scene-specific prompt
            full_prompt = f"{base_prompt}, {scene_prompt}"

            self.log(f"Generating image {i+1}/{len(prompts)}...")

            image_path = output_dir / f"image_{i:03d}.png"

            try:
                await self._generate_image(
                    prompt=full_prompt,
                    output_path=image_path,
                    width=width,
                    height=height,
                )

                results.append(
                    ImageResult(
                        file_path=image_path,
                        prompt=full_prompt,
                        index=i,
                    ))
            except Exception as e:
                self.log(f"Failed to generate image {i}: {e}")

        self.log(f"Generated {len(results)} images")
        return results

    async def _generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1080,
        height: int = 1920,
    ) -> None:
        """Generate single image via Stable Diffusion API"""

        # Automatic1111 / Forge API format
        payload = {
            "prompt": prompt,
            "negative_prompt": self.NEGATIVE_PROMPT,
            "width": width,
            "height": height,
            "steps": 25,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M Karras",
            "seed": -1,  # Random
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.api_url}/sdapi/v1/txt2img",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            image_data = base64.b64decode(data["images"][0])

            # Save image
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path)

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
        """Check if Stable Diffusion API is running"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"{self.api_url}/sdapi/v1/sd-models")
                return response.status_code == 200
        except Exception:
            return False
