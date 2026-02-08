"""
🎙️ Voice Agent - TypeCast TTS (한국 쇼츠 대중 목소리)
개인용 API (https://typecast.ai/developers/api)
"""

from pathlib import Path
from typing import Literal

import httpx

from ..config import settings
from ..models import AudioResult, Script
from .base import BaseAgent

# 콘텐츠 톤별 목소리 매핑
TONE_VOICE_MAP = {
    # 톤: (성별, 연령대, 감정 프리셋)
    "scary": ("male", "young_adult", "normal"),  # 무서운 이야기 → 차분한 남성
    "horror": ("male", "middle_age", "whisper"),  # 공포 → 속삭이는 남성
    "romance": ("female", "young_adult", "happy"),  # 연애 썰 → 밝은 여성
    "funny": ("female", "teenager", "happy"),  # 웃긴 이야기 → 발랄한 10대
    "angry": ("male", "young_adult", "angry"),  # 분노 유발 → 화난 남성
    "sad": ("female", "young_adult", "sad"),  # 슬픈 이야기 → 슬픈 여성
    "news": ("male", "middle_age", "normal"),  # 뉴스/정보 → 차분한 남성
    "gossip": ("female", "young_adult", "toneup"),  # 가십/TMI → 흥분한 여성
    "asmr": ("female", "young_adult", "whisper"),  # ASMR → 속삭이는 여성
    "default": ("female", "young_adult", "smart"),  # 기본 → 여성 스마트 감정
}

ContentTone = Literal["scary", "horror", "romance", "funny", "angry", "sad",
                      "news", "gossip", "asmr", "default"]


class VoiceAgent(BaseAgent[AudioResult]):
    """Agent for generating AI voiceover using TypeCast (Personal API)"""

    # 새 개인용 API 엔드포인트
    API_BASE = "https://api.typecast.ai"

    @property
    def name(self) -> str:
        return "🎙️ VoiceAgent"

    def __init__(self):
        super().__init__()
        self.api_key = settings.tts.typecast_api_key
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        # 캐시된 목소리 목록
        self._voices_cache: list[dict] = []

    async def run(
            self,
            script: Script,
            output_path: Path,
            tone: ContentTone = "default",
            voice_name: str = None,  # 직접 목소리 이름 지정 가능
    ) -> AudioResult:
        """Generate voiceover from script using TypeCast
        
        Args:
            script: 스크립트
            output_path: 저장 경로
            tone: 콘텐츠 톤 (scary, romance, funny 등) - 자동 목소리 매칭
            voice_name: 직접 목소리 이름 지정 (예: "Moonjung")
        """
        self.log("Generating voiceover with TypeCast...")

        if not self.api_key:
            raise ValueError(
                "TypeCast API key not configured (TYPECAST_API_KEY)")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 목소리 목록 캐시
        if not self._voices_cache:
            self._voices_cache = await self.list_voices()
            if not self._voices_cache:
                raise ValueError("No TypeCast voices available")

        # 목소리 선택
        if voice_name:
            # 직접 지정한 경우
            voice_info = self._find_voice_by_name(voice_name)
            emotion = "smart"
        else:
            # 톤에 맞게 자동 매칭
            voice_info, emotion = self._match_voice_by_tone(tone)

        voice_id = voice_info["voice_id"]
        self.log(
            f"Using voice: {voice_info.get('voice_name', voice_id)} (tone: {tone}, emotion: {emotion})"
        )

        # 감정 프롬프트 설정
        if emotion == "smart":
            prompt = {"emotion_type": "smart"}
        else:
            prompt = {"emotion_type": "preset", "emotion_preset": emotion}

        # TTS 생성
        # 쇼츠는 빠른 템포가 좋음 (1.1~1.2배속)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.API_BASE}/v1/text-to-speech",
                headers=self.headers,
                json={
                    "voice_id": voice_id,
                    "text": script.full_text[:2000],  # Max 2000 chars
                    "model": "ssfm-v30",
                    "language": "kor",
                    "prompt": prompt,
                    "output": {
                        "volume": 100,
                        "audio_pitch": 0,
                        "audio_tempo": 1.15,  # 쇼츠용 약간 빠른 속도
                        "audio_format": "mp3",
                    },
                },
            )
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        # Estimate duration (faster tempo)
        char_count = len(script.full_text.replace(" ", ""))
        duration = char_count / 3.5  # 빠른 템포 반영

        self.log(f"Audio saved: {output_path} (~{duration:.1f}s)")

        return AudioResult(
            file_path=output_path,
            duration=duration,
            voice_id=voice_id,
        )

    def _match_voice_by_tone(self, tone: ContentTone) -> tuple[dict, str]:
        """콘텐츠 톤에 맞는 목소리 자동 매칭"""
        gender, age, emotion = TONE_VOICE_MAP.get(tone,
                                                  TONE_VOICE_MAP["default"])

        # 조건에 맞는 목소리 찾기
        candidates = [
            v for v in self._voices_cache
            if v.get("gender") == gender and v.get("age") == age
        ]

        if candidates:
            # TikTok/Reels 우선
            for v in candidates:
                if "Tiktok/Reels" in v.get("use_cases", []):
                    return v, emotion
            return candidates[0], emotion

        # 조건 완화: 성별만 맞춰서 찾기
        candidates = [
            v for v in self._voices_cache if v.get("gender") == gender
        ]
        if candidates:
            return candidates[0], emotion

        # 최후의 수단
        return self._voices_cache[0], emotion

    def _find_voice_by_name(self, name: str) -> dict:
        """이름으로 목소리 찾기"""
        for voice in self._voices_cache:
            if voice.get("voice_name", "").lower() == name.lower():
                return voice

        # 부분 매칭
        for voice in self._voices_cache:
            if name.lower() in voice.get("voice_name", "").lower():
                return voice

        # 못 찾으면 기본값
        self.log(f"Voice '{name}' not found, using default")
        return self._voices_cache[0]

    async def list_voices(self, use_case: str = None) -> list[dict]:
        """List available TypeCast voices"""
        if not self.api_key:
            return []

        try:
            params = {"model": "ssfm-v30"}  # 최신 모델만
            if use_case:
                params["use_cases"] = use_case

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.API_BASE}/v2/voices",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.log(f"Error listing voices: {e}")
            return []

    async def list_voices_for_shorts(self) -> list[dict]:
        """TikTok/Reels/Shorts용 목소리만 조회"""
        return await self.list_voices(use_case="Tiktok/Reels")
