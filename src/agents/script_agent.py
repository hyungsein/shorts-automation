"""
📝 Script Agent - Generates viral scripts for shorts
"""

from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..models import ContentTone, ContentType, Script, TrendData
from .base import BaseAgent

SCRIPT_SYSTEM_PROMPT = """You are a viral YouTube Shorts scriptwriter. Your scripts are:
- HOOK: First 3 seconds MUST grab attention (question, shocking statement, or cliffhanger)
- ENGAGING: Keep viewers watching till the end
- CONCISE: 45-60 seconds when read aloud (about 100-150 words)
- EMOTIONAL: Make viewers feel something

Language: {language}

Rules:
1. Start with a powerful hook - no "Hey guys" or introductions
2. Use simple, conversational language
3. Build tension or curiosity throughout
4. End with a memorable conclusion or call-to-action
5. NO emojis in the script (they can't be read aloud)
"""

SCRIPT_USER_PROMPT = """Create a YouTube Shorts script from this content:

Title: {title}
Source: {source}
Original Content:
{content}

Content Type: {content_type}

Generate a script with:
1. HOOK (first 3 seconds - must grab attention immediately)
2. BODY (main story - keep it engaging)
3. CTA (call to action - "Follow for more" or similar)
4. TONE (choose the best tone for this content)
5. SCENES (15-20 scenes with camera effects - 스토리에 맞는 카메라 워크!)

CAMERA EFFECTS (각 장면 앞에 붙여서 사용):
- [zoom_in] 중요한 순간, 표정 강조, 충격적인 장면
- [zoom_out] 물건→사람, 디테일→전체 상황 보여줄 때
- [pan_left] 두 사람 대화, A에서 B로 시선 이동
- [pan_right] 반대 방향 시선 이동
- [static] 평범한 장면, 빠른 전환

SCENE 작성 규칙:
- 핵심 물건/음식이 나오면: [zoom_out] 물건 클로즈업 → 상황 전체
- 감정 표현: [zoom_in] 얼굴/표정으로 줌인
- 대화/상호작용: [pan_left] 또는 [pan_right]
- 일반 상황: [static]

Available TONE options:
- scary: 무서운 이야기 (차분한 남성 목소리)
- horror: 공포/소름 (속삭이는 남성)
- romance: 연애 썰 (밝은 여성)
- funny: 웃긴 이야기 (발랄한 여성)
- angry: 분노 유발 (화난 남성)
- sad: 슬픈 이야기 (슬픈 여성)
- news: 정보/팩트 (차분한 남성)
- gossip: 가십/TMI (흥분한 여성)
- default: 일반 (여성 스마트 감정)

Output format:
HOOK:
[Your hook here]

BODY:
[Your main content here]

CTA:
[Your call to action here]

TONE:
[Choose one: scary/horror/romance/funny/angry/sad/news/gossip/default]

SCENES:
- [zoom_out] 치킨 클로즈업, 김이 모락모락
- [static] 여자가 치킨 한 조각 집는 모습
- [zoom_in] 맛있게 먹으며 행복한 표정
- [pan_left] 옆에서 부러운 눈으로 쳐다보는 동료
... (15-20 scenes total)
"""


class ScriptAgent(BaseAgent[Script]):
    """Agent for generating viral scripts"""

    @property
    def name(self) -> str:
        return "📝 ScriptAgent"

    async def run(
        self,
        trend: TrendData,
        language: str = "Korean",
        target_duration: float = 45.0,
    ) -> Script:
        """Generate a viral script from trend data"""
        self.log(f"Generating script for: {trend.title[:50]}...")

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SCRIPT_SYSTEM_PROMPT),
            ("user", SCRIPT_USER_PROMPT),
        ])

        # Generate script
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "language":
            language,
            "title":
            trend.title,
            "source":
            trend.source,
            "content":
            trend.content[:3000],  # Limit content length
            "content_type":
            trend.content_type.value,
        })

        # Parse response
        script = self._parse_script(response.content)

        self.log(
            f"Script generated: {len(script.full_text)} chars, {len(script.scene_prompts)} scenes"
        )
        return script

    def _parse_script(self, response: str) -> Script:
        """Parse LLM response into Script object"""
        hook = ""
        body = ""
        cta = ""
        tone_str = "default"
        scene_prompts = []

        current_section = None
        lines = response.split("\n")

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith("HOOK:"):
                current_section = "hook"
                remaining = line[line.upper().find("HOOK:") + 5:].strip()
                if remaining:
                    hook = remaining
            elif line_upper.startswith("BODY:"):
                current_section = "body"
                remaining = line[line.upper().find("BODY:") + 5:].strip()
                if remaining:
                    body = remaining
            elif line_upper.startswith("CTA:"):
                current_section = "cta"
                remaining = line[line.upper().find("CTA:") + 4:].strip()
                if remaining:
                    cta = remaining
            elif line_upper.startswith("TONE:"):
                current_section = "tone"
                remaining = line[line.upper().find("TONE:") +
                                 5:].strip().lower()
                if remaining:
                    tone_str = remaining
            elif line_upper.startswith("SCENES:"):
                current_section = "scenes"
            elif current_section and line.strip():
                if current_section == "hook":
                    hook += " " + line.strip() if hook else line.strip()
                elif current_section == "body":
                    body += " " + line.strip() if body else line.strip()
                elif current_section == "cta":
                    cta += " " + line.strip() if cta else line.strip()
                elif current_section == "tone":
                    tone_str = line.strip().lower()
                elif current_section == "scenes":
                    # Parse scene lines with camera effect
                    # Format: - [effect] description
                    scene_line = line.strip()
                    if scene_line.startswith("-"):
                        scene_line = scene_line[1:].strip()
                    if scene_line:
                        # Extract camera effect [zoom_in], [zoom_out], etc.
                        effect = "static"
                        if scene_line.startswith("["):
                            end_bracket = scene_line.find("]")
                            if end_bracket > 0:
                                effect = scene_line[1:end_bracket].lower()
                                scene_line = scene_line[end_bracket +
                                                        1:].strip()

                        # Remove "Scene X:" prefix if present
                        if ":" in scene_line and scene_line.split(
                                ":")[0].lower().startswith("scene"):
                            scene_line = scene_line.split(":", 1)[1].strip()

                        if scene_line:
                            # Store as "effect|prompt" format
                            scene_prompts.append(f"{effect}|{scene_line}")

        # Convert tone string to enum
        try:
            tone = ContentTone(tone_str)
        except ValueError:
            tone = ContentTone.DEFAULT
            self.log(f"Unknown tone '{tone_str}', using default")

        script = Script(
            hook=hook.strip(),
            body=body.strip(),
            cta=cta.strip(),
            tone=tone,
            scene_prompts=scene_prompts,
        )
        script.combine()

        self.log(f"Detected tone: {tone.value}")
        return script

    async def generate_metadata(
        self,
        script: Script,
        trend: TrendData,
    ) -> dict:
        """Generate YouTube metadata (title, description, tags)"""
        self.log("Generating metadata...")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a YouTube SEO expert. Generate metadata that maximizes views.
            Output in this exact format:
            TITLE: [Catchy title under 60 chars, use hooks like numbers, questions, or shocking words]
            DESCRIPTION: [2-3 sentences with keywords, include call to action]
            TAGS: [comma-separated relevant tags, 10-15 tags]"""),
            ("user", """Generate YouTube Shorts metadata for this script:

            Hook: {hook}
            Content Type: {content_type}
            Source: {source}
            
            Full Script:
            {script}"""),
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "hook": script.hook,
            "content_type": trend.content_type.value,
            "source": trend.source,
            "script": script.full_text,
        })

        return self._parse_metadata(response.content)

    def _parse_metadata(self, response: str) -> dict:
        """Parse metadata from LLM response"""
        title = ""
        description = ""
        tags = []

        for line in response.split("\n"):
            line_upper = line.upper()
            if line_upper.startswith("TITLE:"):
                title = line[6:].strip()
            elif line_upper.startswith("DESCRIPTION:"):
                description = line[12:].strip()
            elif line_upper.startswith("TAGS:"):
                tags_str = line[5:].strip()
                tags = [t.strip() for t in tags_str.split(",")]

        return {
            "title": title,
            "description": description,
            "tags": tags,
        }
