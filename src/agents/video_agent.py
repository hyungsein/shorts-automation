"""
🎬 Video Agent - Creates final short video with subtitles
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)

from ..config import settings
from ..models import AudioResult, Script, VideoResult
from .base import BaseAgent


class VideoAgent(BaseAgent[VideoResult]):
    """Agent for creating short videos with subtitles"""
    
    @property
    def name(self) -> str:
        return "🎬 VideoAgent"
    
    # Short video dimensions (9:16 aspect ratio)
    WIDTH = 1080
    HEIGHT = 1920
    
    async def run(
        self,
        audio: AudioResult,
        script: Script,
        output_path: Path,
        background_video: Optional[Path] = None,
    ) -> VideoResult:
        """Create final short video"""
        self.log("Creating video...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio_clip = AudioFileClip(str(audio.file_path))
        duration = audio_clip.duration
        
        # Create or load background
        if background_video and background_video.exists():
            bg_clip = self._load_background_video(background_video, duration)
        else:
            bg_clip = self._create_gradient_background(duration)
        
        # Generate subtitles
        subtitle_clips = await self._generate_subtitles(script, duration)
        
        # Compose final video
        final_clip = CompositeVideoClip(
            [bg_clip] + subtitle_clips,
            size=(self.WIDTH, self.HEIGHT)
        )
        final_clip = final_clip.set_audio(audio_clip)
        
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
            has_subtitles=True,
        )
    
    def _create_gradient_background(self, duration: float) -> ColorClip:
        """Create a simple dark background"""
        return ColorClip(
            size=(self.WIDTH, self.HEIGHT),
            color=(15, 15, 20),  # Dark blue-ish
            duration=duration,
        )
    
    def _load_background_video(self, video_path: Path, duration: float) -> VideoFileClip:
        """Load and resize background video"""
        clip = VideoFileClip(str(video_path))
        
        # Loop if too short
        if clip.duration < duration:
            loops_needed = int(duration / clip.duration) + 1
            clip = concatenate_videoclips([clip] * loops_needed)
        
        # Trim to exact duration
        clip = clip.subclip(0, duration)
        
        # Resize to fill screen (crop to 9:16)
        clip = clip.resize(height=self.HEIGHT)
        if clip.w < self.WIDTH:
            clip = clip.resize(width=self.WIDTH)
        
        # Center crop
        clip = clip.crop(
            x_center=clip.w / 2,
            y_center=clip.h / 2,
            width=self.WIDTH,
            height=self.HEIGHT,
        )
        
        # Remove original audio
        clip = clip.without_audio()
        
        return clip
    
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
            group = " ".join(words[i:i+4])
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
            txt_clip = txt_clip.set_position(("center", self.HEIGHT * 0.65))
            txt_clip = txt_clip.set_start(start_time)
            txt_clip = txt_clip.set_duration(end_time - start_time)
            
            subtitle_clips.append(txt_clip)
        
        return subtitle_clips
    
    async def transcribe_audio(self, audio_path: Path) -> list[dict]:
        """Transcribe audio using Whisper for accurate subtitles"""
        self.log("Transcribing audio with Whisper...")
        
        try:
            import whisper
            
            model = whisper.load_model("base")
            result = model.transcribe(
                str(audio_path),
                language=settings.default_language,
                word_timestamps=True,
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                })
            
            return segments
            
        except ImportError:
            self.log("Whisper not installed, using text-based subtitles")
            return []
