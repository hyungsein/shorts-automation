"""
🎙️ Voice Agent - TypeCast TTS (한국 쇼츠 대중 목소리)
"""

import asyncio
from pathlib import Path

import httpx

from ..config import settings
from ..models import AudioResult, Script
from .base import BaseAgent


class VoiceAgent(BaseAgent[AudioResult]):
    """Agent for generating AI voiceover using TypeCast"""

    API_BASE = "https://typecast.ai/api"

    @property
    def name(self) -> str:
        return "🎙️ VoiceAgent"

    def __init__(self):
        super().__init__()
        self.api_token = settings.tts.typecast_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # 인기 한국어 목소리 (TypeCast에서 actor_id 확인 후 업데이트)
        # https://biz.typecast.ai 에서 캐릭터 선택 후 API 연동하면 actor_id 확인 가능
        self.voices = {
            # 기본값 - TypeCast 가입 후 /api/actor로 조회해서 업데이트 필요
            "default": None,  # 첫 번째 actor 자동 사용
        }

    async def run(
        self,
        script: Script,
        output_path: Path,
        voice: str = "default",
    ) -> AudioResult:
        """Generate voiceover from script using TypeCast"""
        self.log("Generating voiceover with TypeCast...")

        if not self.api_token:
            raise ValueError(
                "TypeCast API key not configured (TYPECAST_API_KEY)")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get actor_id
        actor_id = self.voices.get(voice)
        if not actor_id:
            # 첫 번째 사용 가능한 actor 가져오기
            actors = await self.list_actors()
            if not actors:
                raise ValueError("No TypeCast actors available")
            actor_id = actors[0]["actor_id"]
            self.log(f"Using actor: {actors[0].get('name', actor_id)}")

        # Step 1: Request speech synthesis
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.API_BASE}/speak",
                headers=self.headers,
                json={
                    "text": script.full_text,
                    "lang": "ko-kr",
                    "actor_id": actor_id,
                    "xapi_hd": True,  # High quality 44.1kHz
                    "xapi_audio_format": "mp3",
                    "model_version": "latest",
                    "speed_x": 1.0,  # Normal speed
                    "max_seconds": 60,  # Max 60 seconds
                },
            )
            response.raise_for_status()
            speak_url = response.json()["result"]["speak_v2_url"]

        # Step 2: Poll until done
        self.log("Waiting for synthesis...")
        audio_url = await self._poll_for_result(speak_url)

        # Step 3: Download audio
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        # Estimate duration (Korean ~3 chars per second)
        char_count = len(script.full_text.replace(" ", ""))
        duration = char_count / 3.0

        self.log(f"Audio saved: {output_path} (~{duration:.1f}s)")

        return AudioResult(
            file_path=output_path,
            duration=duration,
            voice_id=actor_id,
        )

    async def _poll_for_result(self,
                               speak_url: str,
                               max_attempts: int = 60) -> str:
        """Poll TypeCast API until audio is ready"""
        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(max_attempts):
                response = await client.get(speak_url, headers=self.headers)
                response.raise_for_status()

                result = response.json()["result"]
                status = result["status"]

                if status == "done":
                    return result["audio_download_url"]
                elif status == "failed":
                    raise RuntimeError("TypeCast synthesis failed")

                # Wait 1 second before next poll
                await asyncio.sleep(1)

        raise TimeoutError("TypeCast synthesis timed out")

    async def list_actors(self) -> list[dict]:
        """List available TypeCast actors (voices)"""
        if not self.api_token:
            return []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.API_BASE}/actor",
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()["result"]
        except Exception as e:
            self.log(f"Error listing actors: {e}")
            return []

    async def get_account_info(self) -> dict:
        """Get TypeCast account info"""
        if not self.api_token:
            return {}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.API_BASE}/me",
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()["result"]
        except Exception as e:
            self.log(f"Error getting account info: {e}")
            return {}
