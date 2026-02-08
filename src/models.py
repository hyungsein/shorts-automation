"""
📦 Data Models
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content for shorts"""
    AUTO = "auto"  # LLM이 자동 생성 (기본값)
    YOUTUBE_SEARCH = "youtube_search"  # YouTube 검색 참고
    CUSTOM = "custom"  # 직접 주제 입력


class ContentTone(str, Enum):
    """콘텐츠 톤 - 목소리 자동 매칭용"""
    SCARY = "scary"  # 무서운 이야기 → 차분한 남성
    HORROR = "horror"  # 공포 → 속삭이는 남성
    ROMANCE = "romance"  # 연애 썰 → 밝은 여성
    FUNNY = "funny"  # 웃긴 이야기 → 발랄한 여성
    ANGRY = "angry"  # 분노 유발 → 화난 남성
    SAD = "sad"  # 슬픈 이야기 → 슬픈 여성
    NEWS = "news"  # 뉴스/정보 → 차분한 남성
    GOSSIP = "gossip"  # 가십/TMI → 흥분한 여성
    ASMR = "asmr"  # ASMR → 속삭이는 여성
    DEFAULT = "default"  # 기본 → 여성 스마트 감정


class TrendData(BaseModel):
    """Trending topic data"""
    title: str
    source: str
    url: Optional[str] = None
    score: int = 0
    content: str = ""
    content_type: ContentType = ContentType.AUTO
    fetched_at: datetime = Field(default_factory=datetime.now)


class CameraEffect(str, Enum):
    """카메라 효과"""
    ZOOM_IN = "zoom_in"  # 줌인 (감정, 충격)
    ZOOM_OUT = "zoom_out"  # 줌아웃 (물건→전체)
    PAN_LEFT = "pan_left"  # 왼쪽 패닝
    PAN_RIGHT = "pan_right"  # 오른쪽 패닝
    STATIC = "static"  # 정적


class SceneInfo(BaseModel):
    """장면 정보 (프롬프트 + 카메라 효과)"""
    prompt: str
    effect: CameraEffect = CameraEffect.STATIC


class Script(BaseModel):
    """Generated script for a short"""
    hook: str  # First 3 seconds
    body: str  # Main content
    cta: str  # Call to action
    full_text: str = ""

    # 콘텐츠 톤 (목소리 자동 매칭용)
    tone: ContentTone = ContentTone.DEFAULT

    # Scene descriptions for image generation (레거시 호환)
    scene_prompts: list[str] = Field(default_factory=list)

    # Scene with camera effects (새로운 방식)
    scenes: list[SceneInfo] = Field(default_factory=list)

    def combine(self) -> str:
        """Combine all parts into full script"""
        self.full_text = f"{self.hook}\n\n{self.body}\n\n{self.cta}"
        return self.full_text


class ImageResult(BaseModel):
    """Generated image result"""
    file_path: Path
    prompt: str
    index: int = 0


class AudioResult(BaseModel):
    """Generated audio result"""
    file_path: Path
    duration: float
    voice_id: str


class VideoResult(BaseModel):
    """Generated video result"""
    file_path: Path
    duration: float
    resolution: tuple[int, int] = (1080, 1920)
