"""
🎬 Video Agent - Creates final short video with images, TTS, and subtitles
"""

import asyncio
from pathlib import Path

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import MultiplyVolume
from PIL import Image
import random

from ..config import settings
from ..models import AudioResult, ImageResult, Script, VideoResult
from .base import BaseAgent


class VideoAgent(BaseAgent[VideoResult]):
    """Agent for creating short videos with images and subtitles"""

    @property
    def name(self) -> str:
        return "🎬 VideoAgent"

    # Short video dimensions (9:16 aspect ratio)
    WIDTH = 1080
    HEIGHT = 1920

    # BGM 폴더 경로
    BGM_DIR = Path(__file__).parent.parent.parent / "assets" / "bgm"

    def _get_bgm(self) -> Path | None:
        """고정 BGM 반환 (soft_ambient.mp3)"""
        bgm_path = self.BGM_DIR / "soft_ambient.mp3"
        if bgm_path.exists():
            return bgm_path
        return None

    async def run(
            self,
            images: list[ImageResult],
            audio: AudioResult,
            script: Script,
            output_path: Path,
            title: str = None,  # 상단 제목 (선택)
    ) -> VideoResult:
        """Create final short video with images and subtitles"""
        self.log("Creating video...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load TTS audio
        tts_clip = AudioFileClip(str(audio.file_path))
        duration = tts_clip.duration

        # Load BGM (있으면 TTS와 믹스)
        bgm_path = self._get_bgm()
        if bgm_path:
            self.log(f"🎵 BGM: {bgm_path.name}")
            bgm_clip = AudioFileClip(str(bgm_path))
            # BGM을 영상 길이에 맞게 자르기
            if bgm_clip.duration > duration:
                bgm_clip = bgm_clip.subclipped(0, duration)
            else:
                # BGM이 짧으면 루프
                from moviepy import concatenate_audioclips
                loops_needed = int(duration / bgm_clip.duration) + 1
                bgm_clips = [
                    AudioFileClip(str(bgm_path)) for _ in range(loops_needed)
                ]
                bgm_clip = concatenate_audioclips(bgm_clips).subclipped(
                    0, duration)
            # BGM 볼륨 낮추기 (TTS가 메인) - 15%
            bgm_clip = bgm_clip.with_effects([MultiplyVolume(0.15)])
            # TTS + BGM 믹스
            audio_clip = CompositeAudioClip([tts_clip, bgm_clip])
        else:
            self.log("⚠️ No BGM found in assets/bgm/ folder")
            audio_clip = tts_clip

        # Create image slideshow
        if images:
            bg_clip = self._create_image_slideshow(images, duration)
        else:
            bg_clip = self._create_gradient_background(duration)

        # Generate subtitles
        subtitle_clips = await self._generate_subtitles(script, duration)

        # Generate title (상단에 표시)
        title_clips = []
        if title:
            title_clips = self._create_title_clip(title, duration)

        # Compose final video
        final_clip = CompositeVideoClip([bg_clip] + title_clips +
                                        subtitle_clips,
                                        size=(self.WIDTH, self.HEIGHT))
        final_clip = final_clip.with_audio(audio_clip)

        self.log(f"Audio duration: {audio_clip.duration:.1f}s")

        # Export
        self.log(f"Exporting video to {output_path}...")
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            audio=True,  # 오디오 포함 명시
        )

        # Cleanup
        audio_clip.close()
        bg_clip.close()
        final_clip.close()

        self.log(f"Video created: {output_path}")

        return VideoResult(
            file_path=output_path,
            duration=duration,
            resolution=(self.WIDTH, self.HEIGHT),
        )

    def _create_image_slideshow(
        self,
        images: list[ImageResult],
        duration: float,
    ) -> CompositeVideoClip:
        """Create dynamic slideshow with intentional camera effects"""
        if not images:
            return self._create_gradient_background(duration)

        time_per_image = duration / len(images)
        image_clips = []

        for i, img_result in enumerate(images):
            img_path = img_result.file_path

            # Load image
            img_clip = ImageClip(str(img_path))
            img_clip = self._resize_to_fit(img_clip)

            # 프롬프트에서 카메라 효과 추출 (format: "effect|prompt")
            effect_type = "static"
            if "|" in img_result.prompt:
                parts = img_result.prompt.split("|", 1)
                effect_type = parts[0].strip()

            # 유효한 효과인지 확인 (shake 제외 - 어지러움)
            valid_effects = ["zoom_in", "zoom_out", "static", "fade"]
            if effect_type not in valid_effects:
                effect_type = "static"

            # 줌/팬 효과 적용
            img_clip = self._apply_dynamic_effect(img_clip, effect_type,
                                                  time_per_image)

            # Set timing
            img_clip = img_clip.with_start(i * time_per_image)
            img_clip = img_clip.with_duration(time_per_image)

            image_clips.append(img_clip)

        # Create background
        bg = ColorClip(
            size=(self.WIDTH, self.HEIGHT),
            color=(15, 15, 20),
            duration=duration,
        )

        return CompositeVideoClip([bg] + image_clips,
                                  size=(self.WIDTH, self.HEIGHT))

    def _apply_dynamic_effect(
        self,
        clip: ImageClip,
        effect_type: str,
        duration: float,
    ) -> ImageClip:
        """
        다이나믹 효과 적용
        - zoom_in: 천천히 줌인 (강조)
        - zoom_out: 줌아웃 (전체 상황)
        - shake: 화면 흔들림 (충격, 놀람)
        - fade: 페이드 효과 (장면 전환)
        - static: 효과 없음
        """
        import math

        # 줌 범위 (1.0 = 원본, 1.15 = 15% 확대)
        zoom_start = 1.0
        zoom_end = 1.15

        def zoom_in_effect(t):
            """시간에 따라 줌인"""
            progress = t / duration if duration > 0 else 0
            scale = zoom_start + (zoom_end - zoom_start) * progress
            return scale

        def zoom_out_effect(t):
            """시간에 따라 줌아웃"""
            progress = t / duration if duration > 0 else 0
            scale = zoom_end - (zoom_end - zoom_start) * progress
            return scale

        def shake_position(t):
            """화면 흔들림 효과"""
            intensity = 8  # 흔들림 강도 (픽셀)
            frequency = 15  # 흔들림 빈도
            x = int(math.sin(t * frequency) * intensity)
            y = int(math.cos(t * frequency * 1.3) * intensity * 0.5)
            return (x, y)

        if effect_type == "zoom_in":
            return clip.resized(lambda t: zoom_in_effect(t))
        elif effect_type == "zoom_out":
            return clip.resized(lambda t: zoom_out_effect(t))
        elif effect_type == "shake":
            # 흔들림 + 살짝 줌인
            clip = clip.resized(1.05)
            return clip.with_position(shake_position)
        elif effect_type == "fade":
            # 페이드인 효과
            return clip.with_effects([lambda c: c.crossfadein(0.3)])
        else:
            return clip

    def _resize_to_fit(self, clip: ImageClip) -> ImageClip:
        """Resize image clip to fit 9:16 - 화면 꽉 채우고 위아래 크롭"""
        # 화면을 꽉 채우고 2배 확대
        scale_w = self.WIDTH / clip.w
        scale_h = self.HEIGHT / clip.h
        scale = max(scale_w, scale_h) * 2.0  # 2배 확대

        new_w = int(clip.w * scale)
        new_h = int(clip.h * scale)

        clip = clip.resized((new_w, new_h))

        # 중앙 배치 (화면 꽉 채움)
        x_pos = (self.WIDTH - new_w) // 2
        y_pos = (self.HEIGHT - new_h) // 2

        return clip.with_position((x_pos, y_pos))

    def _create_title_clip(self, title: str, duration: float) -> list:
        """상단에 제목 오버레이 (반투명 배경 + 흰색 글씨)"""

        # 제목이 너무 길면 자르기
        if len(title) > 25:
            title = title[:22] + "..."

        title_clips = []

        # 제목 텍스트
        main_title = TextClip(
            text=title,
            font_size=48,
            color="white",
            font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
            method="caption",
            size=(self.WIDTH - 100, None),
            text_align="center",
            stroke_color="black",
            stroke_width=3,
        )

        # 제목 배경 박스 (반투명 검정)
        title_w, title_h = main_title.size
        title_bg = ColorClip(
            size=(title_w + 40, title_h + 20),
            color=(0, 0, 0),
        ).with_opacity(0.6)

        # 상단 배치 (y=80)
        title_bg = title_bg.with_position(("center", 70))
        title_bg = title_bg.with_duration(duration)
        main_title = main_title.with_position(("center", 80))
        main_title = main_title.with_duration(duration)

        title_clips.append(title_bg)
        title_clips.append(main_title)

        return title_clips

    async def _generate_subtitles(
        self,
        script: Script,
        duration: float,
    ) -> list[TextClip]:
        """
        Generate animated subtitle clips - 요즘 쇼츠 스타일
        
        특징:
        - 짧게 짧게 (2-4 단어씩)
        - 빠르게 전환 (답답하지 않게)
        - 하단 safe zone에 배치
        - 큰 글씨 + 테두리 (가독성)
        """
        subtitle_clips = []

        # 스크립트를 문장 단위로 먼저 분리
        text = script.full_text

        # 따옴표 제거 (", ", ‘, ’, “, ” 등)
        text = text.replace('"', '').replace("'", '')
        text = text.replace('“', '').replace('”',
                                             '').replace('‘',
                                                         '').replace('’', '')

        # 마침표, 물음표, 느낌표로 문장 분리
        import re
        sentences = re.split(r'(?<=[.?!])\s+', text)

        # 각 문장을 짧은 구절로 분리 (2-4 단어)
        phrases = []
        for sentence in sentences:
            words = sentence.split()

            # 한국어 특성상 2-3 단어가 적당 (긴 단어 많음)
            chunk_size = 2 if any(len(w) > 5 for w in words) else 3

            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                if chunk:
                    phrase = " ".join(chunk)
                    # 너무 짧은 구절은 다음과 합치기
                    if phrases and len(phrase) < 4 and len(phrases[-1]) < 15:
                        phrases[-1] += " " + phrase
                    else:
                        phrases.append(phrase)

        if not phrases:
            return subtitle_clips

        # 각 구절의 표시 시간 계산
        # 최소 0.4초, 최대 1.5초 (글자수에 비례)
        total_chars = sum(len(p) for p in phrases)

        # 시간 배분
        phrase_times = []
        current_time = 0.0

        for phrase in phrases:
            # 글자수 기반 시간 (한 글자당 약 0.08초, 최소 0.5초)
            char_time = len(phrase) * 0.08
            phrase_duration = max(0.5, min(1.5, char_time))

            phrase_times.append({
                "text": phrase,
                "start": current_time,
                "duration": phrase_duration
            })
            current_time += phrase_duration

        # 전체 시간에 맞게 스케일 조정
        if current_time > 0:
            scale = duration / current_time
            for pt in phrase_times:
                pt["start"] *= scale
                pt["duration"] *= scale

        self.log(f"Creating {len(phrases)} subtitle segments")

        for pt in phrase_times:
            # 자막 텍스트 클립 생성 (두꺼운 글씨 + 테두리)
            txt_clip = TextClip(
                text=pt["text"],
                font_size=72,  # 더 큰 글씨
                color="white",
                font="/System/Library/Fonts/AppleSDGothicNeo.ttc",
                method="caption",
                size=(self.WIDTH - 160, None),
                text_align="center",
                stroke_color="black",  # 검정 테두리
                stroke_width=3,  # 테두리 두께
            )

            # 검정색 배경 박스 생성 (더 크게)
            txt_w, txt_h = txt_clip.size
            padding_x = 40  # 좌우 패딩
            padding_y = 30  # 상하 패딩
            bg_clip = ColorClip(
                size=(txt_w + padding_x * 2, txt_h + padding_y * 2),
                color=(0, 0, 0),  # 검정색
            )

            # 배경 + 텍스트 합치기
            bg_clip = bg_clip.with_position(("center", self.HEIGHT * 0.72))
            txt_clip = txt_clip.with_position(
                ("center", self.HEIGHT * 0.72 + padding_y))

            bg_clip = bg_clip.with_start(pt["start"]).with_duration(
                pt["duration"])
            txt_clip = txt_clip.with_start(pt["start"]).with_duration(
                pt["duration"])

            subtitle_clips.append(bg_clip)
            subtitle_clips.append(txt_clip)

        return subtitle_clips

    def _create_gradient_background(self, duration: float) -> ColorClip:
        """Create a simple dark background"""
        return ColorClip(
            size=(self.WIDTH, self.HEIGHT),
            color=(15, 15, 20),
            duration=duration,
        )
