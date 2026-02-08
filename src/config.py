"""
⚙️ Configuration Management
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class AWSConfig(BaseModel):
    """AWS Bedrock Configuration"""
    access_key_id: str = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    secret_access_key: str = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    region: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", "ap-northeast-2"))
    model_id: str = Field(default_factory=lambda: os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"))


class TTSConfig(BaseModel):
    """TypeCast TTS Configuration (한국 쇼츠 대중 목소리)"""
    typecast_api_key: str = Field(
        default_factory=lambda: os.getenv("TYPECAST_API_KEY", ""))
    default_voice: str = Field(default="default")


class RedditConfig(BaseModel):
    """Reddit API Configuration"""
    client_id: str = Field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    client_secret: str = Field(
        default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    user_agent: str = Field(default_factory=lambda: os.getenv(
        "REDDIT_USER_AGENT", "shorts-automation/1.0"))


class StableDiffusionConfig(BaseModel):
    """Stable Diffusion (Local) Configuration"""
    api_url: str = Field(default_factory=lambda: os.getenv(
        "SD_API_URL", "http://127.0.0.1:7860"))
    model: str = Field(default_factory=lambda: os.getenv("SD_MODEL", ""))


class Settings(BaseModel):
    """Main Settings"""
    # Sub-configs
    aws: AWSConfig = Field(default_factory=AWSConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    sd: StableDiffusionConfig = Field(default_factory=StableDiffusionConfig)

    # General settings
    output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    default_language: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "ko"))

    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


# Global settings instance
settings = Settings()
