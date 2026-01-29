"""
⚙️ Configuration Management
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class AWSConfig(BaseModel):
    """AWS Bedrock Configuration"""
    access_key_id: str = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    secret_access_key: str = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    region: str = Field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    model_id: str = Field(default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


class TTSConfig(BaseModel):
    """Text-to-Speech Configuration"""
    elevenlabs_api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    default_voice: str = Field(default_factory=lambda: os.getenv("DEFAULT_VOICE", "adam"))
    

class RedditConfig(BaseModel):
    """Reddit API Configuration"""
    client_id: str = Field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    user_agent: str = Field(default_factory=lambda: os.getenv("REDDIT_USER_AGENT", "shorts-automation/1.0"))


class YouTubeConfig(BaseModel):
    """YouTube API Configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    client_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET", ""))


class PexelsConfig(BaseModel):
    """Pexels API Configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("PEXELS_API_KEY", ""))


class Settings(BaseModel):
    """Main Settings"""
    # Sub-configs
    aws: AWSConfig = Field(default_factory=AWSConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    pexels: PexelsConfig = Field(default_factory=PexelsConfig)
    
    # General settings
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    shorts_per_day: int = Field(default_factory=lambda: int(os.getenv("SHORTS_PER_DAY", "3")))
    default_language: str = Field(default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "ko"))
    
    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


# Global settings instance
settings = Settings()
