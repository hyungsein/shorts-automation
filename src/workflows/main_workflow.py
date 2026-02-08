"""
🔄 Main Shorts Workflow (with Supervisor Review)
"""

import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..agents import (
    ImageAgent,
    ReviewResult,
    ScriptAgent,
    SupervisorAgent,
    TrendAgent,
    VideoAgent,
    VoiceAgent,
)
from ..config import settings
from ..models import (
    AudioResult,
    ContentType,
    ImageResult,
    Script,
    TrendData,
    VideoResult,
)

# 최대 재시도 횟수
MAX_RETRIES = 3


class WorkflowState(TypedDict):
    """State for the shorts workflow"""
    short_id: str
    content_type: ContentType
    category: str | None
    topic: str | None  # 직접 입력 주제
    search_query: str | None

    # Generated data
    trend: TrendData | None
    trends_pool: list[TrendData]  # 후보 트렌드들
    script: Script | None
    images: list[ImageResult]
    audio: AudioResult | None
    video: VideoResult | None

    # Retry tracking
    trend_attempts: int
    script_attempts: int
    image_attempts: int
    audio_attempts: int

    # Error handling
    error: str | None


class ShortsWorkflow:
    """Main workflow for generating Shorts with Supervisor review"""

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: True면 감독이 승인할 때까지 재시도
        """
        self.trend_agent = TrendAgent()
        self.script_agent = ScriptAgent()
        self.image_agent = ImageAgent()
        self.voice_agent = VoiceAgent()
        self.video_agent = VideoAgent()
        self.supervisor = SupervisorAgent()

        self.strict_mode = strict_mode
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("fetch_trend", self._fetch_trend)
        workflow.add_node("generate_script", self._generate_script)
        workflow.add_node("generate_images", self._generate_images)
        workflow.add_node("generate_audio", self._generate_audio)
        workflow.add_node("final_review", self._final_review)
        workflow.add_node("create_video", self._create_video)

        # Add edges
        workflow.add_edge("fetch_trend", "generate_script")
        workflow.add_edge("generate_script", "generate_images")
        workflow.add_edge("generate_images", "generate_audio")
        workflow.add_edge("generate_audio", "final_review")
        workflow.add_edge("final_review", "create_video")
        workflow.add_edge("create_video", END)

        workflow.set_entry_point("fetch_trend")

        return workflow.compile()

    async def run(
        self,
        content_type: ContentType = ContentType.AUTO,
        category: str = None,
        topic: str = None,
        search_query: str = None,
    ) -> VideoResult | None:
        """Run the full workflow
        
        Args:
            content_type: 콘텐츠 타입 (기본: AUTO)
            category: 카테고리 (인간관계, 연애 등)
            topic: 직접 입력 주제
            search_query: YouTube 검색어
        """
        short_id = str(uuid.uuid4())[:8]

        initial_state: WorkflowState = {
            "short_id": short_id,
            "content_type": content_type,
            "category": category,
            "topic": topic,
            "search_query": search_query,
            "trend": None,
            "trends_pool": [],
            "script": None,
            "images": [],
            "audio": None,
            "video": None,
            "trend_attempts": 0,
            "script_attempts": 0,
            "image_attempts": 0,
            "audio_attempts": 0,
            "error": None,
        }

        mode = "👨‍💼 STRICT" if self.strict_mode else "🚀 FAST"
        print(f"\n{'=' * 60}")
        print(f"🎬 SHORTS AUTOMATION ({mode} MODE)")
        print(f"{'=' * 60}")

        result = await self.graph.ainvoke(initial_state)

        if result.get("error"):
            print(f"\n❌ Workflow failed: {result['error']}")
            return None

        return result.get("video")

    async def _fetch_trend(self, state: WorkflowState) -> WorkflowState:
        """Fetch trending content with supervisor review"""
        print("\n" + "─" * 50)
        print("🔥 Step 1: Fetching Topics...")
        print("─" * 50)

        try:
            # 트렌드 풀이 비어있으면 새로 가져오기
            if not state["trends_pool"]:
                content_type = state["content_type"]

                # 1. CUSTOM: 직접 주제 입력
                if content_type == ContentType.CUSTOM and state.get("topic"):
                    from ..models import TrendData
                    trends = [
                        TrendData(
                            title=state["topic"],
                            source="user_input",
                            score=100.0,
                        )
                    ]
                    print(f"📝 Custom topic: {state['topic']}")

                # 2. YOUTUBE_SEARCH: YouTube 키워드 검색
                elif content_type == ContentType.YOUTUBE_SEARCH and state.get(
                        "search_query"):
                    trends = await self.trend_agent.search_youtube(
                        query=state["search_query"],
                        limit=10,
                    )
                    print(f"🔍 YouTube search: {state['search_query']}")

                # 3. AUTO (기본): LLM 자동 생성
                else:
                    trends = await self.trend_agent.run(
                        category=state.get("category"),
                        count=5,
                    )
                    print(f"🤖 Auto-generated topics")

                if not trends:
                    return {**state, "error": "No topics found"}

                # 점수순 정렬
                trends.sort(key=lambda t: t.score, reverse=True)
                state = {**state, "trends_pool": trends}

            # 가장 높은 점수의 트렌드 선택
            trends_pool = state["trends_pool"]
            if not trends_pool:
                return {
                    **state, "error": "All trends rejected, no more candidates"
                }

            trend = trends_pool[0]
            remaining = trends_pool[1:]

            print(f"📌 Candidate: {trend.title[:50]}...")
            print(f"   Score: {trend.score} | Source: {trend.source}")

            # 감독 평가
            if self.strict_mode:
                feedback = await self.supervisor.review_trend(trend)

                print(f"\n👨‍💼 Supervisor says: {feedback.feedback[:100]}...")

                if feedback.result == ReviewResult.REJECTED:
                    attempts = state["trend_attempts"] + 1
                    print(f"❌ REJECTED (attempt {attempts}/{MAX_RETRIES})")
                    print(
                        f"   Suggestions: {', '.join(feedback.suggestions[:2])}"
                    )

                    if attempts >= MAX_RETRIES:
                        return {
                            **state, "error":
                            f"Supervisor rejected {MAX_RETRIES} trends"
                        }

                    # 다음 트렌드로 재시도
                    return {
                        **state,
                        "trends_pool": remaining,
                        "trend_attempts": attempts,
                    }

            print(f"✅ APPROVED: {trend.title[:40]}...")
            return {**state, "trend": trend, "trends_pool": remaining}

        except Exception as e:
            return {**state, "error": str(e)}

    async def _generate_script(self, state: WorkflowState) -> WorkflowState:
        """Generate script with supervisor review"""
        print("\n" + "─" * 50)
        print("📝 Step 2: Generating Script...")
        print("─" * 50)

        if state.get("error"):
            return state

        attempts = state["script_attempts"]

        while attempts < MAX_RETRIES:
            attempts += 1
            print(f"\n🔄 Attempt {attempts}/{MAX_RETRIES}")

            try:
                script = await self.script_agent.run(trend=state["trend"])

                print(
                    f"   Generated: {len(script.full_text)} chars, {len(script.scene_prompts)} scenes"
                )
                print(f"   Hook: {script.hook[:50]}...")

                # 감독 평가
                if self.strict_mode:
                    feedback = await self.supervisor.review_script(
                        script, state["trend"])

                    print(f"\n👨‍💼 Supervisor (Score: {feedback.score}/10)")
                    print(f"   {feedback.feedback[:100]}...")

                    if feedback.result == ReviewResult.APPROVED:
                        print("✅ APPROVED!")
                        return {
                            **state, "script": script,
                            "script_attempts": attempts
                        }

                    if feedback.result == ReviewResult.REJECTED:
                        print(f"❌ REJECTED")
                        for s in feedback.suggestions[:2]:
                            print(f"   → {s}")
                        continue

                    # NEEDS_REVISION - 한번 더 시도
                    print(f"🔄 NEEDS REVISION")
                    for s in feedback.suggestions[:2]:
                        print(f"   → {s}")
                    continue
                else:
                    return {
                        **state, "script": script,
                        "script_attempts": attempts
                    }

            except Exception as e:
                print(f"   Error: {e}")
                continue

        return {
            **state, "error": f"Script rejected after {MAX_RETRIES} attempts"
        }

    async def _generate_images(self, state: WorkflowState) -> WorkflowState:
        """Generate images with supervisor review"""
        print("\n" + "─" * 50)
        print("🎨 Step 3: Generating Images...")
        print("─" * 50)

        if state.get("error"):
            return state

        attempts = state["image_attempts"]

        while attempts < MAX_RETRIES:
            attempts += 1
            print(f"\n🔄 Attempt {attempts}/{MAX_RETRIES}")

            try:
                output_dir = settings.output_dir / state["short_id"] / "images"
                images = await self.image_agent.run(
                    prompts=state["script"].scene_prompts,
                    output_dir=output_dir,
                )

                print(f"   Generated {len(images)} images")

                # 감독 평가 (프롬프트 기반)
                if self.strict_mode:
                    feedback = await self.supervisor.review_images(
                        images, state["script"])

                    print(f"\n👨‍💼 Supervisor (Score: {feedback.score}/10)")
                    print(f"   {feedback.feedback[:100]}...")

                    if feedback.result == ReviewResult.APPROVED:
                        print("✅ APPROVED!")
                        return {
                            **state, "images": images,
                            "image_attempts": attempts
                        }

                    # 이미지는 비용이 많이 드니 낮은 기준으로 통과
                    if feedback.score >= 6:
                        print("✅ ACCEPTABLE (score >= 6)")
                        return {
                            **state, "images": images,
                            "image_attempts": attempts
                        }

                    print(f"❌ REJECTED")
                    continue
                else:
                    return {
                        **state, "images": images,
                        "image_attempts": attempts
                    }

            except Exception as e:
                print(f"   Error: {e}")
                continue

        return {
            **state, "error": f"Images rejected after {MAX_RETRIES} attempts"
        }

    async def _generate_audio(self, state: WorkflowState) -> WorkflowState:
        """Generate TTS audio with supervisor review"""
        print("\n" + "─" * 50)
        print("🎙️ Step 4: Generating Audio...")
        print("─" * 50)

        if state.get("error"):
            return state

        try:
            output_path = settings.output_dir / state["short_id"] / "audio.mp3"

            # 스크립트의 톤에 맞는 목소리 자동 매칭
            script = state["script"]
            tone = script.tone.value if hasattr(script, 'tone') else "default"
            print(f"   Content tone: {tone}")

            audio = await self.voice_agent.run(
                script=script,
                output_path=output_path,
                tone=tone,  # 톤에 맞는 목소리 자동 선택
            )

            print(f"   Duration: {audio.duration:.1f}s")

            # 감독 평가
            if self.strict_mode:
                feedback = await self.supervisor.review_audio(
                    audio, state["script"])

                print(f"\n👨‍💼 Supervisor (Score: {feedback.score}/10)")
                print(f"   {feedback.feedback[:100]}...")

                # 오디오는 재생성이 어려우니 경고만
                if feedback.result == ReviewResult.REJECTED:
                    print("⚠️ Warning: Audio not ideal, but proceeding...")

            print(f"✅ Audio ready: {audio.duration:.1f}s")
            return {**state, "audio": audio}

        except Exception as e:
            return {**state, "error": str(e)}

    async def _final_review(self, state: WorkflowState) -> WorkflowState:
        """Final supervisor review before video creation"""
        print("\n" + "─" * 50)
        print("👨‍💼 Step 5: FINAL SUPERVISOR REVIEW")
        print("─" * 50)

        if state.get("error"):
            return state

        if not self.strict_mode:
            print("⏭️ Strict mode OFF - skipping final review")
            return state

        feedback = await self.supervisor.final_review(
            trend=state["trend"],
            script=state["script"],
            image_count=len(state["images"]),
            audio_duration=state["audio"].duration,
        )

        print(f"\n{'='*50}")
        print(f"👨‍💼 FINAL VERDICT: {feedback.score}/10")
        print(f"{'='*50}")
        print(f"\n{feedback.feedback}")

        if feedback.suggestions:
            print("\n📋 Notes:")
            for s in feedback.suggestions:
                print(f"   • {s}")

        if feedback.result == ReviewResult.REJECTED:
            print("\n❌ FINAL REVIEW FAILED")
            print("   The supervisor has rejected this Short.")
            return {
                **state, "error":
                "Final review failed - content not up to standards"
            }

        print("\n✅ APPROVED FOR VIDEO CREATION!")
        return state

    async def _create_video(self, state: WorkflowState) -> WorkflowState:
        """Create final video"""
        print("\n" + "─" * 50)
        print("🎬 Step 6: Creating Video...")
        print("─" * 50)

        if state.get("error"):
            return state

        try:
            output_path = settings.output_dir / state["short_id"] / "final.mp4"
            video = await self.video_agent.run(
                images=state["images"],
                audio=state["audio"],
                script=state["script"],
                output_path=output_path,
            )

            print(f"\n{'='*60}")
            print(f"🎉 VIDEO COMPLETE!")
            print(f"{'='*60}")
            print(f"📁 Output: {video.file_path}")
            print(f"⏱️ Duration: {video.duration:.1f}s")
            print(f"📐 Resolution: {video.resolution[0]}x{video.resolution[1]}")

            return {**state, "video": video}

        except Exception as e:
            return {**state, "error": str(e)}
