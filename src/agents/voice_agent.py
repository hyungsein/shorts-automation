"""
🎙️ Voice Agent - Generates AI voiceover using TTS
"""

import asyncio
from pathlib import Path
from typing import Optional

from elevenlabs import AsyncElevenLabs
from elevenlabs.core import ApiError

from ..config import settings
from ..models import AudioResult, Script
from .base import BaseAgent


class VoiceAgent(BaseAgent[AudioResult]):
    """Agent for generating AI voiceover"""
    
    @property
    def name(self) -> str:
        return "🎙️ VoiceAgent"
    
    def __init__(self):
        super().__init__()
        self.client: Optional[AsyncElevenLabs] = None
        self._init_client()
        
        # Popular voice IDs (ElevenLabs)
        self.voices = {
            "adam": "pNInz6obpgDQGcFmaJgB",      # Deep male
            "rachel": "21m00Tcm4TlvDq8ikWAM",    # Female
            "domi": "AZnzlk1XvdvUeBnXmlld",      # Female energetic
            "bella": "EXAVITQu4vr4xnSDxMaL",     # Female soft
            "antoni": "ErXwobaYiN019PkySvjV",    # Male warm
            "josh": "TxGEqnHWrfWFTfGW9XjX",      # Male young
            "arnold": "VR6AewLTigWG4xSOukaG",    # Male deep
            "sam": "yoZ06aMxZJJ28mfd3POQ",       # Male neutral
        }
    
    def _init_client(self) -> None:
        """Initialize ElevenLabs client"""
        if settings.tts.elevenlabs_api_key:
            self.client = AsyncElevenLabs(
                api_key=settings.tts.elevenlabs_api_key
            )
    
    async def run(
        self,
        script: Script,
        output_path: Path,
        voice: str = "adam",
    ) -> AudioResult:
        """Generate voiceover from script"""
        self.log(f"Generating voiceover with voice: {voice}")
        
        if not self.client:
            raise ValueError("ElevenLabs API key not configured")
        
        # Get voice ID
        voice_id = self.voices.get(voice, voice)
        
        # Generate audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            audio = await self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=script.full_text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            
            # Save audio file
            audio_bytes = b""
            async for chunk in audio:
                audio_bytes += chunk
            
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            # Get duration (estimate based on text length)
            # Average speaking rate: ~150 words per minute
            word_count = len(script.full_text.split())
            duration = (word_count / 150) * 60
            
            self.log(f"Audio saved: {output_path} ({duration:.1f}s)")
            
            return AudioResult(
                file_path=output_path,
                duration=duration,
                voice_id=voice_id,
            )
            
        except ApiError as e:
            self.log(f"ElevenLabs API error: {e}")
            raise
    
    async def list_voices(self) -> list[dict]:
        """List available voices"""
        if not self.client:
            return []
        
        try:
            response = await self.client.voices.get_all()
            return [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "category": v.category,
                }
                for v in response.voices
            ]
        except Exception as e:
            self.log(f"Error listing voices: {e}")
            return []
    
    async def get_remaining_characters(self) -> int:
        """Get remaining character quota"""
        if not self.client:
            return 0
        
        try:
            user = await self.client.user.get()
            subscription = user.subscription
            return subscription.character_limit - subscription.character_count
        except Exception as e:
            self.log(f"Error getting quota: {e}")
            return 0
