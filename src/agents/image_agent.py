"""
🎨 Image Agent - Generates images using diffusers (local, no server needed)
"""

import asyncio
import random
from pathlib import Path
from typing import Optional

import httpx
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from ..models import ImageResult
from .base import BaseAgent


class ImageAgent(BaseAgent[list[ImageResult]]):
    """Agent for generating images with Stable Diffusion (diffusers)"""

    # Model path - MeinaMix V11 (귀여운 로맨스 스타일)
    MODEL_PATH = Path.home(
    ) / "ComfyUI" / "models" / "checkpoints" / "meinamix_v11.safetensors"

    # HuggingFace fallback model
    HF_MODEL = "Meina/MeinaMix_V11"

    @property
    def name(self) -> str:
        return "🎨 ImageAgent"

    # 쇼츠용 캐릭터 스타일 - 글래머 오피스 여캐
    CHARACTER_BASE_PROMPT = """
    masterpiece, best quality, beautiful detailed eyes,
    1girl, office lady, business suit, pencil skirt,
    large breasts, slim waist, attractive body,
    pretty face, makeup, long hair,
    soft lighting, clean background
    """.strip()

    NEGATIVE_PROMPT = """
    ugly, deformed, noisy, blurry, low quality,
    bad anatomy, bad proportions, watermark, text,
    realistic, photo, nsfw, nude, naked, explicit,
    extra fingers, mutated hands, poorly drawn face,
    flat chest, child, loli, underage
    """.strip()

    def __init__(self):
        super().__init__()
        self._pipe: Optional[StableDiffusionPipeline] = None

    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load the Stable Diffusion pipeline (lazy loading)"""
        if self._pipe is not None:
            return self._pipe

        self.log("Loading MeinaMix model (first time may take a while)...")

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            device = "mps"
            # MPS에서는 float32가 더 안정적
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
            self._pipe = StableDiffusionPipeline.from_single_file(
                str(self.MODEL_PATH),
                torch_dtype=dtype,
                use_safetensors=True,
            )
        else:
            self.log(
                f"Local model not found, downloading from HuggingFace: {self.HF_MODEL}"
            )
            self._pipe = StableDiffusionPipeline.from_pretrained(
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
            width: int = 512,  # SD 1.5 기본 해상도
            height: int = 768,  # 세로로 길게 (쇼츠용)
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
            # 카메라 효과와 프롬프트 분리 (format: "effect|prompt")
            effect = "static"
            actual_prompt = scene_prompt
            if "|" in scene_prompt:
                parts = scene_prompt.split("|", 1)
                effect = parts[0].strip()
                actual_prompt = parts[1].strip()

            # Combine character base + scene-specific prompt
            full_prompt = f"{base_prompt}, {actual_prompt}"

            self.log(f"Generating image {i+1}/{len(prompts)} [{effect}]...")

            image_path = output_dir / f"image_{i:03d}.png"

            try:
                # Run generation in thread pool (sync -> async)
                image = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._generate_sync(pipe, full_prompt, width,
                                                      height))

                # Resize to shorts format (9:16) - 1080x1920
                shorts_image = self._resize_for_shorts(image)
                shorts_image.save(image_path)

                # 효과 정보를 프롬프트 앞에 유지 (video_agent에서 사용)
                results.append(
                    ImageResult(
                        file_path=image_path,
                        prompt=f"{effect}|{actual_prompt}",
                        index=i,
                    ))
                self.log(f"✓ Image {i+1} saved")
            except Exception as e:
                self.log(f"Failed to generate image {i}: {e}")

        self.log(f"Generated {len(results)} images")
        return results

    def _generate_sync(
        self,
        pipe: StableDiffusionPipeline,
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
        """
        Resize image for YouTube Shorts with safe zone consideration.
        
        YouTube Shorts UI overlay:
        - Top ~15%: 채널명, 팔로우 버튼, 검색 등
        - Bottom ~20%: 좋아요, 댓글, 공유, 자막 영역
        
        전략: 1024x1024 이미지를 중앙에 배치하고 위아래에 블러 배경 추가
        """
        target_w, target_h = 1080, 1920

        # 1. 이미지를 target_w에 맞게 리사이즈 (비율 유지)
        img_w, img_h = image.size
        scale = target_w / img_w
        new_w = target_w
        new_h = int(img_h * scale)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 2. 새 캔버스 생성 (1080x1920)
        # 배경: 이미지 가장자리 색상 기반 그라데이션 효과
        canvas = Image.new('RGB', (target_w, target_h), (20, 20, 25))

        # 3. 블러된 배경 이미지 생성 (위아래 채우기용)
        from PIL import ImageFilter

        # 이미지를 전체 캔버스 크기로 늘려서 블러 (배경용)
        bg_image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        bg_blurred = bg_image.filter(ImageFilter.GaussianBlur(radius=30))

        # 블러 배경을 어둡게 (자막 가독성)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(bg_blurred)
        bg_darkened = enhancer.enhance(0.4)  # 40% 밝기

        canvas.paste(bg_darkened, (0, 0))

        # 4. 메인 이미지를 중앙보다 약간 위에 배치 (하단 자막 공간 확보)
        # 상단 15%, 하단 20% = safe zone 밖
        # 이미지를 약간 위로 올려서 하단에 자막 공간 확보

        top_margin = int(target_h * 0.12)  # 상단 12% 여백
        bottom_margin = int(target_h * 0.22)  # 하단 22% 여백 (자막 + UI)

        available_height = target_h - top_margin - bottom_margin

        if new_h > available_height:
            # 이미지가 safe zone보다 크면 축소
            scale = available_height / new_h
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
            resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 중앙 정렬 (수평), safe zone 내 중앙 (수직)
        x = (target_w - new_w) // 2
        y = top_margin + (available_height - new_h) // 2

        canvas.paste(resized, (x, y))

        return canvas

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

    async def search_and_download_image(
        self,
        query: str,
        output_path: Path,
    ) -> Optional[Path]:
        """
        Unsplash에서 무료 이미지 검색 & 다운로드
        주제에 맞는 실제 이미지 (은수저, 카페, 헬스장 등)
        """
        self.log(f"Searching image for: {query}")

        try:
            # Unsplash API (무료, API 키 불필요한 방식)
            async with httpx.AsyncClient(timeout=30) as client:
                # 검색 URL (source.unsplash.com 리다이렉트 사용)
                search_url = f"https://source.unsplash.com/800x600/?{query}"

                response = await client.get(search_url, follow_redirects=True)

                if response.status_code == 200:
                    # 이미지 저장
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    # 쇼츠 포맷으로 리사이즈
                    img = Image.open(output_path)
                    resized = self._resize_for_shorts(img)
                    resized.save(output_path)

                    self.log(f"✓ Downloaded: {query}")
                    return output_path

        except Exception as e:
            self.log(f"Failed to search image: {e}")

        return None

    async def get_topic_image(
        self,
        topic: str,
        output_dir: Path,
    ) -> Optional[ImageResult]:
        """
        주제에 맞는 대표 이미지 가져오기
        예: "은수저" → 은수저 이미지, "카페" → 카페 이미지
        """
        # 한국어 → 영어 키워드 매핑 (검색용)
        keyword_map = {
            "은수저": "silver spoon wealth",
            "금수저": "gold spoon luxury",
            "흙수저": "poor struggle",
            "카페": "coffee shop barista",
            "헬스장": "gym fitness",
            "회사": "office workplace",
            "직장": "corporate office",
            "알바": "part time job",
            "연애": "couple love",
            "썸": "romantic dating",
            "친구": "friendship friends",
            "가족": "family",
            "학교": "school student",
            "대학": "university college",
            "면접": "job interview",
            "이직": "career change",
            "퇴사": "quit job resignation",
            "월급": "salary paycheck money",
            "부자": "rich wealthy luxury",
            "여행": "travel vacation",
        }

        # 주제에서 키워드 추출
        search_query = None
        for korean, english in keyword_map.items():
            if korean in topic:
                search_query = english
                break

        if not search_query:
            # 매핑 없으면 주제 그대로 사용
            search_query = topic

        output_path = output_dir / "topic_image.png"
        result = await self.search_and_download_image(search_query,
                                                      output_path)

        if result:
            return ImageResult(
                file_path=result,
                prompt=f"searched: {search_query}",
                index=-1,  # 특별 이미지 표시
            )

        return None

    async def check_connection(self) -> bool:
        """Check if model is available (local file or can download)"""
        if self.MODEL_PATH.exists():
            return True
        # Can always download from HuggingFace
        return True
