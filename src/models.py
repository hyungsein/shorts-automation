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
    NEWS = "news"
    HISTORY = "history"


class ShortStatus(str, Enum):
    """Status of a short video"""
    PENDING = "pending"
    SCRIPT_READY = "script_ready"
    AUDIO_READY = "audio_ready"
    VIDEO_READY = "video_ready"
    UPLOADED = "uploaded"
    FAILED = "failed"


class TrendData(BaseModel):
    """Trending topic data"""
    title: str
    source: str
    url: Optional[str] = None
    score: int = 0  # upvotes, likes, etc.
    content: str = ""
    content_type: ContentType = ContentType.REDDIT_STORY
    fetched_at: datetime = Field(default_factory=datetime.now)


class Script(BaseModel):
    """Generated script for a short"""
    hook: str  # First 3 seconds - grab attention
    body: str  # Main content
    cta: str  # Call to action at the end
    full_text: str = ""  # Combined text for TTS
    duration_estimate: float = 45.0  # seconds
    
    def combine(self) -> str:
        """Combine all parts into full script"""
        self.full_text = f"{self.hook}\n\n{self.body}\n\n{self.cta}"
        return self.full_text


class AudioResult(BaseModel):
    """Generated audio result"""
    file_path: Path
    duration: float  # seconds
    voice_id: str


class VideoResult(BaseModel):
    """Generated video result"""
    file_path: Path
    duration: float
    resolution: tuple[int, int] = (1080, 1920)  # 9:16 aspect ratio
    has_subtitles: bool = True


class ShortVideo(BaseModel):
    """Complete short video data"""
    id: str
    status: ShortStatus = ShortStatus.PENDING
    
    # Content
    trend: Optional[TrendData] = None
    script: Optional[Script] = None
    
    # Generated files
    audio: Optional[AudioResult] = None
    video: Optional[VideoResult] = None
    
    # YouTube metadata
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    uploaded_at: Optional[datetime] = None
    
    # YouTube info (after upload)
    youtube_id: Optional[str] = None
    youtube_url: Optional[str] = None
