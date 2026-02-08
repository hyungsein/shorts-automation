"""
📝 Script Agent - Generates viral scripts for shorts
"""

from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..models import ContentTone, ContentType, Script, TrendData
from .base import BaseAgent

SCRIPT_SYSTEM_PROMPT = """You are a viral YouTube Shorts scriptwriter who creates ADDICTIVE, jaw-dropping stories. Your scripts go VIRAL because:

- HOOK: 첫 3초에 "뭐?!" 하게 만드는 충격적인 문장 (질문, 반전, 믿기 힘든 사실)
- TWIST: 중간에 "에이 설마..." 하다가 "진짜?!" 하게 만드는 반전
- ENDING: 마지막에 소름돋거나 터지거나 "헐..." 하게 만드는 결말

Language: {language}

=== TTS 자연스러운 문장 연결 (가장 중요!) ===

1. 문장 끝 어미 통일하기 (같은 톤 유지):
   - 친근한 톤: "~거든요", "~잖아요", "~인 거예요", "~더라고요"
   - 전달 톤: "~했어요", "~봤어요", "~됐어요"
   
2. 연결어로 자연스럽게 이어주기:
   - "그래서" → 결과 연결
   - "근데" → 반전/전환
   - "그러다가" → 시간 흐름
   - "그랬더니" → 반응/결과
   - "알고 보니" → 반전 사실
   
3. 문장 사이 호흡 만들기:
   - 쉼표(,)로 짧은 쉼
   - 마침표(.)로 긴 쉼
   - "요" 어미로 끝나면 다음 문장과 자연스럽게 연결됨

=== 자연스러운 대화체 예시 ===
❌ 어색한 연결:
"저 회사 다녔어요. 상사가 있었어요. 그 상사가 저한테 뭐라고 했어요."

✅ 자연스러운 연결:
"저 예전에 회사 다녔거든요. 근데 상사가 좀 이상했어요. 어느 날 저한테 갑자기 뭐라고 하는 거예요."

❌ 어색한 연결:
"남자친구랑 데이트했어요. 카페 갔어요. 핸드폰을 봤어요."

✅ 자연스러운 연결:
"남자친구랑 카페에서 데이트하고 있었거든요. 근데 잠깐 화장실 간 사이에요. 그 남자 핸드폰을 봤는데요."



VIRAL SCRIPT SECRETS:
1. 첫 문장 = 가장 충격적인 부분 먼저!
2. "근데 진짜 소름돋는 건요.", "알고 보니까요." 같은 긴장감 유발
3. 구체적인 디테일 (금액, 시간, 장소 - 한글로!)
4. 감정 폭발 포인트
5. 열린 결말이나 충격적 반전
6. NO "팔로우", "구독", "좋아요"

TONE OPTIONS:
- 무서운 썰 (scary): 소름, 미스터리
- 연애 썰 (romance): 설렘, 배신
- 빡치는 썰 (angry): 진상, 갑질
- 웃긴 썰 (funny): 황당, 민망
- 감동 썰 (sad): 눈물, 이별

EXAMPLE HOOKS (자연스러운 TTS):
- "제가 예전에 회사 다녔거든요. 근데 어느 날 화장실에서요, 대표님 통화를 들었는데요. 제 이름이 나오더라고요."
- "남자친구랑 소개팅으로 만났거든요. 근데 사귀고 나서요, 이상한 점을 발견했어요."
"""

SCRIPT_USER_PROMPT = """Create a VIRAL YouTube Shorts script from this content:

Title: {title}
Source: {source}
Original Content:
{content}

Content Type: {content_type}

Generate a script with:
1. HOOK (첫 3초 - 스크롤 멈추게 만드는 충격적인 첫 문장)
2. BODY (본문 - "근데요", "알고 보니까요", "그런데 진짜 소름돋는 건요" 로 긴장감 유지)
3. CTA (엔딩 - 충격 반전 or 열린 결말 - 절대 "팔로우/구독" 금지)
4. TONE (콘텐츠에 맞는 톤 선택)
5. SCENES (15-20개 장면 + 카메라 효과)



VIRAL WRITING STYLE:
- 실제 경험담처럼 1인칭: "제가요", "저는요", "저한테요"
- 구어체 필수: "~했거든요", "~인 거예요", "~잖아요"
- 감정 표현: "소름 돋았어요", "눈물 났어요", "너무 빡쳤어요"
- 반전 예고: "근데요, 여기서 반전이요.", "알고 보니까요."

CAMERA EFFECTS:
- [zoom_in] 충격 순간, 중요한 대사, 반전 포인트
- [zoom_out] 상황 전체 보여주기
- [static] 일반 대화, 설명
- [fade] 시간 경과, 회상

Output format:
HOOK:
[충격적인 첫 문장 - TTS가 자연스럽게 읽을 수 있게!]

BODY:
[긴장감 있는 본문 - 짧은 문장, 자연스러운 호흡]

CTA:
[충격 반전 or 열린 결말 - 댓글 유도]

TONE:
[scary/horror/romance/funny/angry/sad/news/gossip/default 중 하나]

SCENES (MUST BE IN ENGLISH for image generation):
- [zoom_in] shocked face looking at phone
- [static] couple sitting at cafe
- [fade] girl sitting alone in room
... (15-20 scenes total)

IMPORTANT: SCENES must be in ENGLISH (simple image descriptions).
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
                    # TONE은 한 단어만! (scary, horror, romance 등)
                    tone_str = remaining.split()[0] if remaining.split(
                    ) else "default"
                    current_section = None  # TONE 이후 바로 다음 섹션으로
            elif line_upper.startswith("SCENES"):
                current_section = "scenes"
            elif current_section and line.strip():
                if current_section == "hook":
                    hook += " " + line.strip() if hook else line.strip()
                elif current_section == "body":
                    body += " " + line.strip() if body else line.strip()
                elif current_section == "cta":
                    cta += " " + line.strip() if cta else line.strip()
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
