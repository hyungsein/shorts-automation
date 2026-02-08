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
        """Create slideshow from generated images"""
        if not images:
            return self._create_gradient_background(duration)

        time_per_image = duration / len(images)
        image_clips = []

        for i, img_result in enumerate(images):
            img_path = img_result.file_path

            # Load and resize image to fit 9:16
            img_clip = ImageClip(str(img_path))
            img_clip = self._resize_to_fit(img_clip)

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

    def _resize_to_fit(self, clip: ImageClip) -> ImageClip:
        """Resize image clip to fit 9:16 while maintaining aspect ratio"""
        # Calculate scale to fill the frame
        scale_w = self.WIDTH / clip.w
        scale_h = self.HEIGHT / clip.h
        scale = max(scale_w, scale_h)

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
        """Generate animated subtitle clips"""
        subtitle_clips = []

        # Split script into sentences/phrases
        text = script.full_text
        words = text.split()

        # Group words (3-5 words per subtitle)
        word_groups = []
        for i in range(0, len(words), 4):
            group = " ".join(words[i:i + 4])
            word_groups.append(group)

        if not word_groups:
            return subtitle_clips

        # Calculate timing for each group
        time_per_group = duration / len(word_groups)

        for i, group in enumerate(word_groups):
            start_time = i * time_per_group
            end_time = (i + 1) * time_per_group

            # Create text clip
            txt_clip = TextClip(
                group,
                fontsize=60,
                color="white",
                stroke_color="black",
                stroke_width=3,
                font="Arial-Bold",
                method="caption",
                size=(self.WIDTH - 100, None),
                align="center",
            )

            # Position at bottom third
            txt_clip = txt_clip.with_position(("center", self.HEIGHT * 0.65))
            txt_clip = txt_clip.with_start(start_time)
            txt_clip = txt_clip.with_duration(end_time - start_time)

            subtitle_clips.append(txt_clip)

        return subtitle_clips

    def _create_gradient_background(self, duration: float) -> ColorClip:
        """Create a simple dark background"""
        return ColorClip(
            size=(self.WIDTH, self.HEIGHT),
            color=(15, 15, 20),
            duration=duration,
        )
