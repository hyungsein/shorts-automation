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
    REDDIT_STORY = "reddit_story"
    SCARY_STORY = "scary_story"
    FUN_FACTS = "fun_facts"
    MOTIVATION = "motivation"


class TrendData(BaseModel):
    """Trending topic data"""
    title: str
    source: str
    url: Optional[str] = None
    score: int = 0
    content: str = ""
    content_type: ContentType = ContentType.REDDIT_STORY
    fetched_at: datetime = Field(default_factory=datetime.now)


class Script(BaseModel):
    """Generated script for a short"""
    hook: str  # First 3 seconds
    body: str  # Main content
    cta: str  # Call to action
    full_text: str = ""

    # Scene descriptions for image generation
    scene_prompts: list[str] = Field(default_factory=list)

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
