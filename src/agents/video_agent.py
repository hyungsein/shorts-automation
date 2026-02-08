"""
🎬 Video Agent - Creates final short video with images, TTS, and subtitles
"""

import asyncio
from pathlib import Path

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from PIL import Image

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

    async def run(
        self,
        images: list[ImageResult],
        audio: AudioResult,
        script: Script,
        output_path: Path,
    ) -> VideoResult:
        """Create final short video with images and subtitles"""
        self.log("Creating video...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio_clip = AudioFileClip(str(audio.file_path))
        duration = audio_clip.duration

        # Create image slideshow
        if images:
            bg_clip = self._create_image_slideshow(images, duration)
        else:
            bg_clip = self._create_gradient_background(duration)

        # Generate subtitles
        subtitle_clips = await self._generate_subtitles(script, duration)

        # Compose final video
        final_clip = CompositeVideoClip([bg_clip] + subtitle_clips,
                                        size=(self.WIDTH, self.HEIGHT))
        final_clip = final_clip.with_audio(audio_clip)

        # Export
        self.log(f"Exporting video to {output_path}...")
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
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

            # 유효한 효과인지 확인
            valid_effects = [
                "zoom_in", "zoom_out", "pan_left", "pan_right", "static",
                "zoom_pulse"
            ]
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
        다이나믹 줌/팬 효과 적용
        - zoom_in: 천천히 줌인
        - zoom_out: 줌아웃
        - pan_left/right: 좌우 패닝
        - zoom_pulse: 줌인했다 줌아웃
        """

        # 줌 범위 (1.0 = 원본, 1.2 = 20% 확대)
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

        def zoom_pulse_effect(t):
            """줌인했다가 줌아웃"""
            import math
            progress = t / duration if duration > 0 else 0
            # 사인 곡선으로 부드럽게
            scale = zoom_start + (zoom_end - zoom_start) * math.sin(
                progress * math.pi)
            return scale

        if effect_type == "zoom_in":
            return clip.resized(lambda t: zoom_in_effect(t))
        elif effect_type == "zoom_out":
            return clip.resized(lambda t: zoom_out_effect(t))
        elif effect_type == "zoom_pulse":
            return clip.resized(lambda t: zoom_pulse_effect(t))
        elif effect_type == "pan_left":
            # 오른쪽에서 왼쪽으로 이동
            def pan_left_pos(t):
                progress = t / duration if duration > 0 else 0
                x = int(50 - 100 * progress)  # 50 -> -50
                return (x, 'center')

            return clip.with_position(pan_left_pos)
        elif effect_type == "pan_right":
            # 왼쪽에서 오른쪽으로 이동
            def pan_right_pos(t):
                progress = t / duration if duration > 0 else 0
                x = int(-50 + 100 * progress)  # -50 -> 50
                return (x, 'center')

            return clip.with_position(pan_right_pos)
        else:
            return clip

    def _resize_to_fit(self, clip: ImageClip) -> ImageClip:
        """Resize image clip to fit 9:16 with extra margin for zoom effects"""
        # Calculate scale to fill the frame (줌 효과를 위해 20% 더 크게)
        scale_w = self.WIDTH / clip.w
        scale_h = self.HEIGHT / clip.h
        scale = max(scale_w, scale_h) * 1.2  # 20% 여유

        new_w = int(clip.w * scale)
        new_h = int(clip.h * scale)

        clip = clip.resized((new_w, new_h))

        # Center the clip
        x_pos = (self.WIDTH - new_w) // 2
        y_pos = (self.HEIGHT - new_h) // 2

        return clip.with_position((x_pos, y_pos))

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
            # 자막 클립 생성
            txt_clip = TextClip(
                text=pt["text"],
                font_size=72,  # 더 큰 글씨
                color="white",
                stroke_color="black",
                stroke_width=4,  # 더 두꺼운 테두리
                font="Arial-Bold",
                method="caption",
                size=(self.WIDTH - 120, None),
                text_align="center",
            )

            # 하단 safe zone에 배치 (UI 피해서)
            # 화면의 약 70% 위치 (하단 30% 영역 중 위쪽)
            txt_clip = txt_clip.with_position(("center", self.HEIGHT * 0.68))
            txt_clip = txt_clip.with_start(pt["start"])
            txt_clip = txt_clip.with_duration(pt["duration"])

            subtitle_clips.append(txt_clip)

        return subtitle_clips

    def _create_gradient_background(self, duration: float) -> ColorClip:
        """Create a simple dark background"""
        return ColorClip(
            size=(self.WIDTH, self.HEIGHT),
            color=(15, 15, 20),
            duration=duration,
        )
