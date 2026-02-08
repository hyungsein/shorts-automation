"""
👨‍💼 Supervisor Agent - 깐깐한 감독님
각 Agent의 결과물을 평가하고 OK 사인을 내림
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from ..models import AudioResult, ImageResult, Script, TrendData
from .base import BaseAgent


class ReviewResult(str, Enum):
    """평가 결과"""
    APPROVED = "approved"  # ✅ 통과
    REJECTED = "rejected"  # ❌ 다시해
    NEEDS_REVISION = "revision"  # 🔄 수정 필요


@dataclass
class SupervisorFeedback:
    """감독 피드백"""
    result: ReviewResult
    score: int  # 1-10
    feedback: str
    suggestions: list[str]


SUPERVISOR_SYSTEM_PROMPT = """You are a STRICT and DEMANDING creative director for viral YouTube Shorts.
You have extremely high standards and rarely approve on the first try.
Your job is to review each stage of content creation and give brutally honest feedback.

You are known for:
- Being incredibly picky about quality
- Demanding viral-worthy content only
- Rejecting mediocre work without hesitation
- Providing specific, actionable feedback

Scoring criteria (1-10):
- 1-4: REJECTED - Completely unacceptable, start over
- 5-6: NEEDS_REVISION - Has potential but needs significant changes  
- 7-8: Good but could be better, minor revisions
- 9-10: APPROVED - Excellent, meets viral standards

You must output in this EXACT format:
RESULT: [approved/rejected/revision]
SCORE: [1-10]
FEEDBACK: [Your brutally honest assessment]
SUGGESTIONS:
- [Specific improvement 1]
- [Specific improvement 2]
- [Specific improvement 3]
"""


class SupervisorAgent(BaseAgent[SupervisorFeedback]):
    """깐깐한 감독 Agent - 품질 평가 및 승인"""

    @property
    def name(self) -> str:
        return "👨‍💼 Supervisor"

    async def review_trend(self, trend: TrendData) -> SupervisorFeedback:
        """트렌드 평가 - 바이럴 가능성 체크"""
        self.log(f"Reviewing trend: {trend.title[:30]}...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user",
             """Review this trending content for viral YouTube Shorts potential:

Title: {title}
Source: {source}
Score/Upvotes: {score}
Content Preview:
{content}

Evaluate:
1. Is this topic interesting enough for viral Shorts?
2. Does it have emotional hook potential?
3. Is the story/content compelling?
4. Will Gen Z / young adults engage with this?
5. Can this be condensed into 45-60 seconds effectively?

Be VERY strict. We only want content that has VIRAL potential."""),
        ])

        return await self._get_review(
            prompt, {
                "title": trend.title,
                "source": trend.source,
                "score": trend.score,
                "content": trend.content[:2000],
            })

    async def review_script(self, script: Script,
                            trend: TrendData) -> SupervisorFeedback:
        """스크립트 평가 - 훅, 구성, 바이럴성 체크"""
        self.log("Reviewing script...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user", """Review this YouTube Shorts script:

Original Topic: {title}

=== HOOK (First 3 seconds) ===
{hook}

=== BODY ===
{body}

=== CALL TO ACTION ===
{cta}

=== SCENE PROMPTS ({scene_count} scenes) ===
{scenes}

Evaluate HARSHLY:
1. HOOK: Does it INSTANTLY grab attention? Would you stop scrolling?
2. PACING: Is every second engaging? Any boring parts?
3. EMOTION: Does it make viewers FEEL something?
4. CTA: Is it natural, not cringy?
5. SCENES: Are the visual descriptions vivid and appropriate?
6. LENGTH: Is it concise enough for Shorts? (100-150 words ideal)

Word count: {word_count} words

Reject if:
- Hook is weak or generic
- Body is boring or too long
- Scenes don't match the story
- Overall not viral-worthy"""),
        ])

        scenes_text = "\n".join([f"- {s}" for s in script.scene_prompts])

        return await self._get_review(
            prompt, {
                "title": trend.title,
                "hook": script.hook,
                "body": script.body,
                "cta": script.cta,
                "scenes": scenes_text,
                "scene_count": len(script.scene_prompts),
                "word_count": len(script.full_text.split()),
            })

    async def review_images(self, images: list[ImageResult],
                            script: Script) -> SupervisorFeedback:
        """이미지 프롬프트 평가"""
        self.log(f"Reviewing {len(images)} images...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user", """Review the image generation results:

Script Hook: {hook}
Script Body: {body}

Generated Images ({count} total):
{image_prompts}

Evaluate:
1. Do the images match the story/script?
2. Are the prompts descriptive enough for good visuals?
3. Is there visual variety across scenes?
4. Will these images keep viewers engaged?
5. Is the count appropriate? (4-6 images ideal for 45-60s)

Note: I can only see the prompts, not the actual images.
Judge based on whether the prompts would generate appropriate visuals."""),
        ])

        prompts_text = "\n".join([
            f"Image {i+1}: {img.prompt[:200]}..."
            for i, img in enumerate(images)
        ])

        return await self._get_review(
            prompt, {
                "hook": script.hook,
                "body": script.body[:500],
                "count": len(images),
                "image_prompts": prompts_text,
            })

    async def review_audio(self, audio: AudioResult,
                           script: Script) -> SupervisorFeedback:
        """오디오 평가 - 길이 적절성"""
        self.log(f"Reviewing audio ({audio.duration:.1f}s)...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("user", """Review the TTS audio generation:

Script length: {word_count} words
Audio duration: {duration:.1f} seconds
Voice ID: {voice_id}

Target duration for Shorts: 45-60 seconds

Script content:
{script_text}

Evaluate:
1. Is the duration appropriate for YouTube Shorts? (under 60s)
2. Is the pacing likely to be good? (~150 words/minute is natural)
3. Is the script length appropriate for the content?

Guidelines:
- Under 30s: Too short, might feel rushed
- 30-45s: Acceptable for quick content
- 45-60s: Ideal range
- Over 60s: Too long for Shorts"""),
        ])

        return await self._get_review(
            prompt, {
                "word_count": len(script.full_text.split()),
                "duration": audio.duration,
                "voice_id": audio.voice_id,
                "script_text": script.full_text[:1000],
            })

    async def final_review(
        self,
        trend: TrendData,
        script: Script,
        image_count: int,
        audio_duration: float,
    ) -> SupervisorFeedback:
        """최종 검토 - 전체 패키지 평가"""
        self.log("Final review before video creation...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_SYSTEM_PROMPT + """

This is the FINAL review before video creation.
Be extra critical - once approved, the video will be created."""),
            ("user", """FINAL REVIEW - All components ready for video creation:

=== SOURCE ===
Title: {title}
Source: {source}
Original Score: {trend_score}

=== SCRIPT ===
Hook: {hook}
Body: {body}
CTA: {cta}
Total words: {word_count}

=== ASSETS ===
Images: {image_count} scenes
Audio: {duration:.1f} seconds

=== FINAL CHECKLIST ===
1. ✓ Trending topic selected
2. ✓ Script written with hook, body, CTA
3. ✓ Scene images generated
4. ✓ TTS audio created

Give your FINAL verdict:
- Is this ready to become a viral YouTube Short?
- Any last-minute concerns?
- Overall quality assessment?

Remember: This is your last chance to reject before video creation."""),
        ])

        return await self._get_review(
            prompt, {
                "title": trend.title,
                "source": trend.source,
                "trend_score": trend.score,
                "hook": script.hook,
                "body": script.body,
                "cta": script.cta,
                "word_count": len(script.full_text.split()),
                "image_count": image_count,
                "duration": audio_duration,
            })

    async def _get_review(self, prompt: ChatPromptTemplate,
                          variables: dict) -> SupervisorFeedback:
        """LLM에게 평가 요청"""
        chain = prompt | self.llm
        response = await chain.ainvoke(variables)

        return self._parse_feedback(response.content)

    def _parse_feedback(self, response: str) -> SupervisorFeedback:
        """LLM 응답 파싱"""
        result = ReviewResult.REJECTED
        score = 5
        feedback = ""
        suggestions = []

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith("RESULT:"):
                result_text = line.split(":", 1)[1].strip().lower()
                if "approved" in result_text:
                    result = ReviewResult.APPROVED
                elif "revision" in result_text:
                    result = ReviewResult.NEEDS_REVISION
                else:
                    result = ReviewResult.REJECTED

            elif line_upper.startswith("SCORE:"):
                try:
                    score = int(line.split(":", 1)[1].strip().split()[0])
                    score = max(1, min(10, score))  # Clamp 1-10
                except:
                    score = 5

            elif line_upper.startswith("FEEDBACK:"):
                feedback = line.split(":", 1)[1].strip()
                current_section = "feedback"

            elif line_upper.startswith("SUGGESTIONS:"):
                current_section = "suggestions"

            elif current_section == "feedback" and line.strip():
                if not line.strip().startswith("-"):
                    feedback += " " + line.strip()

            elif current_section == "suggestions" and line.strip():
                if line.strip().startswith("-"):
                    suggestions.append(line.strip()[1:].strip())

        # 점수 기반 자동 결과 조정
        if score >= 9:
            result = ReviewResult.APPROVED
        elif score <= 4:
            result = ReviewResult.REJECTED

        emoji = "✅" if result == ReviewResult.APPROVED else "❌" if result == ReviewResult.REJECTED else "🔄"
        self.log(f"{emoji} Score: {score}/10 - {result.value}")

        return SupervisorFeedback(
            result=result,
            score=score,
            feedback=feedback,
            suggestions=suggestions,
        )
