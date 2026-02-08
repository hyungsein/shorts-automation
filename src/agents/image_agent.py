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

    # Model path - MeinaMix V11 (글래머 오피스 스타일)
    MODEL_PATH = Path.home(
    ) / "ComfyUI" / "models" / "checkpoints" / "meinamix_v11.safetensors"

    # HuggingFace fallback model
    HF_MODEL = "Meina/MeinaMix_V11"

    @property
    def name(self) -> str:
        return "🎨 ImageAgent"

    # 의상 + 배경 매칭 (캐주얼 위주)
    OUTFIT_BACKGROUND_PAIRS = [
        # 👖 청반바지/데님 (잘 뽑히는 스타일!)
        ("crop top, denim shorts",
         ["park, sunny day", "city street, summer", "ice cream shop"]),
        ("white t-shirt, denim shorts",
         ["cafe interior", "convenience store", "street, shopping"]),
        ("tank top, denim shorts",
         ["beach, summer", "pool party", "outdoor cafe"]),
        ("off-shoulder top, denim shorts",
         ["city street", "rooftop, sunny", "park, picnic"]),
        ("striped shirt, denim shorts",
         ["cafe terrace", "bookstore", "street, walking"]),

        # 👕 캐주얼 일상
        ("oversized t-shirt, shorts",
         ["room interior, bedroom", "living room, sofa", "convenience store"]),
        ("hoodie, mini skirt, sneakers",
         ["arcade, game center", "subway station", "street, night"]),
        ("casual dress, sneakers",
         ["park, sunny", "shopping mall", "cafe interior"]),
        ("cardigan, shorts, casual",
         ["cafe interior", "library", "street, autumn"]),

        # 🌞 여름 캐주얼
        ("sleeveless top, hot pants",
         ["beach, sunset", "pool, summer", "rooftop, sunny"]),
        ("sundress, summer dress",
         ["flower field", "beach boardwalk", "outdoor cafe"]),
        ("bikini top, denim shorts",
         ["beach, ocean", "pool party", "resort, summer"]),
    ]

    # 주인공 외모 옵션 (영상 시작 시 랜덤 선택 후 고정)
    # 모든 캐릭터 검은 머리로 통일
    HAIR_OPTIONS = [
        "long straight black hair",  # 긴 생머리
        "short black hair, bob cut",  # 단발
    ]

    FACE_OPTIONS = [
        "pretty face, makeup, black eyes",
        "beautiful face, light makeup, black eyes",
        "cute face, natural makeup, black eyes",
    ]

    # 글래머 캐릭터 (주인공 = {protagonist}) - 짧게!
    CHARACTER_TEMPLATES = [
        # 주인공 혼자
        ("1girl, {protagonist}, {outfit}", 60),
        # 주인공 + 다른 여자
        ("2girls, {protagonist}, another girl, {outfit}", 15),
        # 주인공 + 남자
        ("1boy 1girl, {protagonist}, handsome man", 15),
        # 클로즈업
        ("1girl, {protagonist}, upper body, face focus", 10),
    ]

    # 프롬프트 (간결하게 - CLIP 77토큰 제한)
    QUALITY_PROMPT = "masterpiece, best quality, korean webtoon"

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
        self._protagonist: Optional[str] = None  # 주인공 캐릭터 (영상마다 고정)
        self._protagonist_seed: Optional[int] = None  # 주인공 seed (일관성)

    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load the Stable Diffusion pipeline (lazy loading)"""
        if self._pipe is not None:
            return self._pipe

        self.log(
            "Loading Counterfeit V3 model (first time may take a while)...")

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

    def _create_protagonist(self) -> str:
        """영상 시작 시 주인공 외모 생성 (한 번만) - 짧게!"""
        hair = random.choice(self.HAIR_OPTIONS)
        # 간결하게: 머리 + 몸매만
        protagonist = f"{hair}, pretty face, large breasts"
        self.log(f"🎭 주인공: {hair}")
        return protagonist

    def _pick_outfit_and_background(self) -> tuple[str, str]:
        """의상과 어울리는 배경을 함께 선택"""
        outfit, backgrounds = random.choice(self.OUTFIT_BACKGROUND_PAIRS)
        background = random.choice(backgrounds)
        return outfit, background

    def _pick_character_template(self, scene_prompt: str = "") -> str:
        """씬 내용에 맞는 캐릭터 템플릿 선택 - 씬 프롬프트가 주인공일 때만 캐릭터 추가"""
        scene_lower = scene_prompt.lower()

        # 씬 내용 분석해서 캐릭터 구성 결정
        has_man = any(word in scene_lower for word in [
            "man", "boy", "guy", "boyfriend", "husband", "male", "he ", "him",
            "his", "couple"
        ])
        has_two_girls = any(word in scene_lower for word in [
            "two girls", "2 girls", "friends", "girls talking", "both girls",
            "2girls"
        ])

        # 씬에 이미 의상/직업이 있는지 확인
        has_outfit_in_scene = any(word in scene_lower for word in [
            "uniform", "dress", "outfit", "wearing", "clothes", "suit",
            "attendant", "nurse", "maid", "teacher", "student", "office",
            "bikini", "swimsuit", "pajamas", "coat", "jacket"
        ])

        # 캐릭터 구성만 결정 (의상은 씬에서 가져옴)
        if has_man:
            # 남자가 나오는 씬
            char = f"1boy 1girl, {self._protagonist}, handsome man"
        elif has_two_girls:
            # 여자 둘
            char = f"2girls, {self._protagonist}, another girl"
        else:
            # 기본 1girl
            char = f"1girl, {self._protagonist}"

        # 씬에 의상이 없으면 랜덤 의상 추가
        if not has_outfit_in_scene:
            outfit, _ = self._pick_outfit_and_background()
            char = f"{char}, {outfit}"

        return char

    async def run(
            self,
            prompts: list[str],
            output_dir: Path,
            character_prompt: Optional[str] = None,
            width: int = 512,  # SD 1.5 해상도
            height: int = 680,  # 더 크롭되게 (위아래 많이 잘림)
    ) -> list[ImageResult]:
        """Generate multiple images for the video"""

        self.log(f"Generating {len(prompts)} images...")

        # 🎭 영상마다 주인공 캐릭터 새로 생성 (이 영상 내에서는 고정)
        self._protagonist = self._create_protagonist()
        self._protagonist_seed = random.randint(1, 999999)
        self.log(f"🎲 주인공 seed: {self._protagonist_seed}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

            # 씬 내용 분석해서 적절한 캐릭터 구성 선택
            if character_prompt:
                char_prompt = character_prompt
            else:
                char_prompt = self._pick_character_template(actual_prompt)

            # 프롬프트 순서: 씬 내용 > 캐릭터 > 퀄리티 (CLIP은 앞부분 우선)
            full_prompt = f"{actual_prompt}, {char_prompt}, {self.QUALITY_PROMPT}"

            self.log(f"Generating image {i+1}/{len(prompts)} [{effect}]...")
            self.log(f"  📝 Scene: {actual_prompt}")
            self.log(f"  🎨 Full prompt: {full_prompt[:100]}...")

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
        use_protagonist_seed: bool = True,
    ) -> Image.Image:
        """Synchronous image generation (called in thread pool)"""
        # 주인공이 나오는 씬은 같은 seed 사용 (일관성)
        generator = None
        if use_protagonist_seed and self._protagonist_seed:
            # seed에 약간의 변화를 줘서 완전 똑같진 않게
            seed = self._protagonist_seed + random.randint(0, 100)
            generator = torch.Generator().manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator,
        )
        return result.images[0]

    def _resize_for_shorts(self, image: Image.Image) -> Image.Image:
        """
        Resize image for YouTube Shorts - 가로 꽉 채우고 위아래 자르기
        """
        target_w, target_h = 1080, 1920

        img_w, img_h = image.size

        # 가로를 꽉 채우고 위아래 crop
        scale = target_w / img_w
        new_w = target_w
        new_h = int(img_h * scale)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 위아래 자르기 (중앙 기준)
        if new_h > target_h:
            # 위아래 crop
            top = (new_h - target_h) // 2
            cropped = resized.crop((0, top, target_w, top + target_h))
        else:
            # 세로가 부족하면 검은 배경에 중앙 배치
            canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            y = (target_h - new_h) // 2
            canvas.paste(resized, (0, y))
            cropped = canvas

        return cropped

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
                index=-1,
            )

        return None
