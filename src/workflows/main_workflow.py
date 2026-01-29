"""
🔄 Main Shorts Automation Workflow using LangGraph
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..agents import (
    ScriptAgent,
    TrendAgent,
    UploadAgent,
    VideoAgent,
    VoiceAgent,
)
from ..config import settings
from ..models import (
    AudioResult,
    ContentType,
    Script,
    ShortStatus,
    ShortVideo,
    TrendData,
    VideoResult,
)


class WorkflowState(TypedDict):
    """State for the shorts workflow"""
    # Current short being processed
    short_id: str
    status: ShortStatus
    
    # Content
    content_type: ContentType
    trend: TrendData | None
    script: Script | None
    
    # Generated files
    audio: AudioResult | None
    video: VideoResult | None
    
    # Metadata
    title: str
    description: str
    tags: list[str]
    
    # Results
    youtube_id: str | None
    error: str | None


class ShortsWorkflow:
    """Main workflow for generating YouTube Shorts"""
    
    def __init__(self):
        self.trend_agent = TrendAgent()
        self.script_agent = ScriptAgent()
        self.voice_agent = VoiceAgent()
        self.video_agent = VideoAgent()
        self.upload_agent = UploadAgent()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("fetch_trend", self._fetch_trend)
        workflow.add_node("generate_script", self._generate_script)
        workflow.add_node("generate_audio", self._generate_audio)
        workflow.add_node("create_video", self._create_video)
        workflow.add_node("upload", self._upload)
        
        # Add edges
        workflow.add_edge("fetch_trend", "generate_script")
        workflow.add_edge("generate_script", "generate_audio")
        workflow.add_edge("generate_audio", "create_video")
        workflow.add_edge("create_video", "upload")
        workflow.add_edge("upload", END)
        
        # Set entry point
        workflow.set_entry_point("fetch_trend")
        
        return workflow.compile()
    
    async def _fetch_trend(self, state: WorkflowState) -> WorkflowState:
        """Fetch trending content"""
        print(f"\n{'='*50}")
        print("🔥 Step 1: Fetching Trends")
        print(f"{'='*50}")
        
        try:
            trends = await self.trend_agent.run(
                content_type=state["content_type"],
                limit=5,
            )
            
            if not trends:
                return {**state, "error": "No trends found", "status": ShortStatus.FAILED}
            
            # Pick the top trending content
            trend = max(trends, key=lambda t: t.score)
            print(f"✅ Selected: {trend.title[:50]}... (score: {trend.score})")
            
            return {**state, "trend": trend, "status": ShortStatus.PENDING}
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {**state, "error": str(e), "status": ShortStatus.FAILED}
    
    async def _generate_script(self, state: WorkflowState) -> WorkflowState:
        """Generate script from trend"""
        print(f"\n{'='*50}")
        print("📝 Step 2: Generating Script")
        print(f"{'='*50}")
        
        if state.get("error"):
            return state
        
        try:
            script = await self.script_agent.run(
                trend=state["trend"],
                language="Korean",
                target_duration=45.0,
            )
            
            # Generate metadata
            metadata = await self.script_agent.generate_metadata(
                script=script,
                trend=state["trend"],
            )
            
            print(f"✅ Script generated ({len(script.full_text)} chars)")
            print(f"📌 Title: {metadata['title']}")
            
            return {
                **state,
                "script": script,
                "title": metadata["title"],
                "description": metadata["description"],
                "tags": metadata["tags"],
                "status": ShortStatus.SCRIPT_READY,
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {**state, "error": str(e), "status": ShortStatus.FAILED}
    
    async def _generate_audio(self, state: WorkflowState) -> WorkflowState:
        """Generate voiceover"""
        print(f"\n{'='*50}")
        print("🎙️ Step 3: Generating Audio")
        print(f"{'='*50}")
        
        if state.get("error"):
            return state
        
        try:
            output_dir = settings.ensure_output_dir()
            audio_path = output_dir / f"{state['short_id']}_audio.mp3"
            
            audio = await self.voice_agent.run(
                script=state["script"],
                output_path=audio_path,
                voice=settings.tts.default_voice,
            )
            
            print(f"✅ Audio generated: {audio.file_path}")
            
            return {**state, "audio": audio, "status": ShortStatus.AUDIO_READY}
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {**state, "error": str(e), "status": ShortStatus.FAILED}
    
    async def _create_video(self, state: WorkflowState) -> WorkflowState:
        """Create final video"""
        print(f"\n{'='*50}")
        print("🎬 Step 4: Creating Video")
        print(f"{'='*50}")
        
        if state.get("error"):
            return state
        
        try:
            output_dir = settings.ensure_output_dir()
            video_path = output_dir / f"{state['short_id']}_final.mp4"
            
            video = await self.video_agent.run(
                audio=state["audio"],
                script=state["script"],
                output_path=video_path,
            )
            
            print(f"✅ Video created: {video.file_path}")
            
            return {**state, "video": video, "status": ShortStatus.VIDEO_READY}
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {**state, "error": str(e), "status": ShortStatus.FAILED}
    
    async def _upload(self, state: WorkflowState) -> WorkflowState:
        """Upload to YouTube"""
        print(f"\n{'='*50}")
        print("📤 Step 5: Uploading to YouTube")
        print(f"{'='*50}")
        
        if state.get("error"):
            return state
        
        try:
            # Note: Upload requires OAuth authentication
            # For now, we'll skip actual upload and mark as ready
            print("⚠️ Upload skipped (OAuth not configured)")
            print(f"📁 Video ready at: {state['video'].file_path}")
            
            return {**state, "status": ShortStatus.VIDEO_READY}
            
            # Uncomment below when OAuth is configured:
            # youtube_id = await self.upload_agent.run(
            #     video=state["video"],
            #     title=state["title"],
            #     description=state["description"],
            #     tags=state["tags"],
            # )
            # return {**state, "youtube_id": youtube_id, "status": ShortStatus.UPLOADED}
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {**state, "error": str(e), "status": ShortStatus.FAILED}
    
    async def run(
        self,
        content_type: ContentType = ContentType.REDDIT_STORY,
    ) -> ShortVideo:
        """Execute the full workflow"""
        short_id = str(uuid.uuid4())[:8]
        
        print(f"\n{'='*60}")
        print(f"🎬 Starting Shorts Automation - ID: {short_id}")
        print(f"📂 Content Type: {content_type.value}")
        print(f"{'='*60}")
        
        # Initial state
        initial_state: WorkflowState = {
            "short_id": short_id,
            "status": ShortStatus.PENDING,
            "content_type": content_type,
            "trend": None,
            "script": None,
            "audio": None,
            "video": None,
            "title": "",
            "description": "",
            "tags": [],
            "youtube_id": None,
            "error": None,
        }
        
        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Create ShortVideo result
        short = ShortVideo(
            id=short_id,
            status=final_state["status"],
            trend=final_state.get("trend"),
            script=final_state.get("script"),
            audio=final_state.get("audio"),
            video=final_state.get("video"),
            title=final_state.get("title", ""),
            description=final_state.get("description", ""),
            tags=final_state.get("tags", []),
            youtube_id=final_state.get("youtube_id"),
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Workflow Complete!")
        print(f"📊 Status: {short.status.value}")
        if short.video:
            print(f"📁 Output: {short.video.file_path}")
        print(f"{'='*60}\n")
        
        return short
