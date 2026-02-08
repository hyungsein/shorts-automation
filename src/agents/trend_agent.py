"""
🔥 Trend Agent - LLM 기반 바이럴 주제 자동 생성
"""

import random
from typing import Optional

import httpx
from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..models import ContentType, TrendData
from .base import BaseAgent

# 쇼츠에서 잘 먹히는 카테고리
VIRAL_CATEGORIES = [
    "인간관계",  # 친구 손절, 연인 특징, 가족 갈등
    "직장생활",  # 회사 썰, 상사 유형, 퇴사 썰
    "연애",  # 소개팅, 이별, 썸
    "심리",  # 행동심리, 성격 분석, 숨은 의미
    "공감",  # MZ 공감, 직장인 공감, 학생 공감
    "레전드썰",  # 커뮤니티 레전드, 실화 썰
    "꿀팁",  # 생활 꿀팁, 돈 버는 법
    "충격",  # 충격적인 사실, 반전 이야기
]


class TrendAgent(BaseAgent[list[TrendData]]):
    """LLM 기반 바이럴 주제 자동 생성 에이전트"""

    API_BASE = "https://www.googleapis.com/youtube/v3"

    @property
    def name(self) -> str:
        return "🔥 TrendAgent"

    def __init__(self):
        super().__init__()
        self.api_key = settings.youtube.api_key
        self.region = settings.youtube.region_code

    async def run(
        self,
        content_type: ContentType = ContentType.AUTO,
        limit: int = 5,
        category: str = None,
        topic: str = None,
    ) -> list[TrendData]:
        """바이럴 주제 자동 생성
        
        Args:
            content_type: 콘텐츠 타입 (기본: AUTO)
            limit: 생성할 주제 수
            category: 특정 카테고리 지정 (없으면 랜덤)
            topic: 직접 주제 입력 (있으면 이걸로 바로 사용)
        """

        # 직접 주제 입력한 경우
        if topic:
            self.log(f"Using custom topic: {topic}")
            return [
                TrendData(
                    title=topic,
                    source="custom",
                    content=topic,
                    score=100,
                    content_type=ContentType.CUSTOM,
                )
            ]

        # 카테고리 선택 (없으면 랜덤)
        if not category:
            category = random.choice(VIRAL_CATEGORIES)

        self.log(f"Generating viral topics for category: {category}")

        # YouTube 트렌드 참고 (API 있으면)
        youtube_context = ""
        if self.api_key:
            keywords = await self._get_trending_keywords()
            if keywords:
                youtube_context = f"\n\n참고 - 현재 YouTube 인기 키워드: {', '.join(keywords[:5])}"

        # LLM으로 주제 생성
        topics = await self._generate_viral_topics(category, limit,
                                                   youtube_context)

        self.log(f"Generated {len(topics)} viral topics")
        return topics

    async def _generate_viral_topics(
        self,
        category: str,
        limit: int,
        youtube_context: str = "",
    ) -> list[TrendData]:
        """LLM으로 바이럴 주제 생성"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국 유튜브 쇼츠 바이럴 전문가입니다.
사람들이 클릭하고 끝까지 보게 만드는 주제를 생성합니다.

좋은 쇼츠 주제 특징:
1. 강한 감정 유발 (공감, 분노, 놀람, 웃음)
2. 호기심 자극하는 제목
3. 1분 안에 핵심 전달 가능
4. 한국인이 공감할 수 있는 내용
5. 댓글 달고 싶게 만드는 주제

예시:
- "이런 친구는 지금 당장 손절해라"
- "회사에서 절대 하면 안 되는 행동 TOP 5"
- "소개팅에서 이 말 하면 100% 차인다"
- "월급 200 받으면서 깨달은 것들"

각 주제마다 간단한 내용 요약도 함께 작성하세요."""),
            ("user", """카테고리: {category}
{youtube_context}

위 카테고리에서 바이럴될 쇼츠 주제 {limit}개를 생성하세요.

출력 형식:
1. [제목]
내용: [1-2문장 요약]

2. [제목]
내용: [1-2문장 요약]

..."""),
        ])

        try:
            chain = prompt | self.llm
            self.log("Calling LLM...")
            response = await chain.ainvoke({
                "category": category,
                "youtube_context": youtube_context,
                "limit": limit,
            })
            self.log(f"LLM Response received: {len(response.content)} chars")
            self.log(f"Response preview: {response.content[:200]}...")

            # 파싱
            topics = self._parse_topics(response.content, category)
            self.log(f"Parsed {len(topics)} topics")
            return topics[:limit]
        except Exception as e:
            import traceback
            self.log(f"LLM Error: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return []

    def _parse_topics(self, response: str, category: str) -> list[TrendData]:
        """LLM 응답 파싱"""
        topics = []
        lines = response.strip().split("\n")

        current_title = None
        current_content = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ## 1. 또는 1. 로 시작하는 제목 라인
            # "## 1." 또는 "1." 패턴 감지
            clean_line = line.lstrip("#").strip()

            if clean_line and clean_line[0].isdigit(
            ) and "." in clean_line[:4]:
                # 이전 주제 저장
                if current_title:
                    topics.append(
                        TrendData(
                            title=current_title,
                            source=f"AI생성 ({category})",
                            content=current_content or current_title,
                            score=random.randint(80, 100),
                            content_type=ContentType.AUTO,
                        ))

                # 새 주제 시작 - "1. " 이후 부분 추출
                current_title = clean_line.split(".", 1)[1].strip()
                # 따옴표, 대괄호 제거
                current_title = current_title.strip("[]\"'")
                current_content = ""

            # 내용 라인 - **내용:** 또는 내용: 패턴
            elif "내용:" in line or "요약:" in line:
                # **내용:** 형식 처리
                content_part = line.split(":",
                                          1)[1].strip() if ":" in line else ""
                current_content = content_part.strip("*")
                current_content = line.split(":", 1)[1].strip()

        # 마지막 주제 저장
        if current_title:
            topics.append(
                TrendData(
                    title=current_title,
                    source=f"AI생성 ({category})",
                    content=current_content or current_title,
                    score=random.randint(80, 100),
                    content_type=ContentType.AUTO,
                ))

        return topics

    async def _get_trending_keywords(self) -> list[str]:
        """YouTube 인기 영상에서 키워드 추출 (참고용)"""
        if not self.api_key:
            return []

        params = {
            "part": "snippet",
            "chart": "mostPopular",
            "regionCode": self.region,
            "maxResults": 10,
            "key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.API_BASE}/videos",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                keywords = []
                for item in data.get("items", []):
                    title = item.get("snippet", {}).get("title", "")
                    # 간단히 제목에서 키워드 추출
                    keywords.append(title[:30])

                return keywords

        except Exception as e:
            self.log(f"YouTube API error (ignored): {e}")
            return []

    async def search_youtube(self,
                             query: str,
                             limit: int = 5) -> list[TrendData]:
        """YouTube 검색 (옵션)"""
        if not self.api_key:
            return []

        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "viewCount",
            "regionCode": self.region,
            "maxResults": limit,
            "key": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.API_BASE}/search",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()

                trends = []
                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    video_id = item.get("id", {}).get("videoId", "")

                    trends.append(
                        TrendData(
                            title=snippet.get("title", ""),
                            source=
                            f"YouTube ({snippet.get('channelTitle', '')})",
                            url=f"https://youtube.com/watch?v={video_id}",
                            score=0,
                            content=snippet.get("description", ""),
                            content_type=ContentType.YOUTUBE_SEARCH,
                        ))

                return trends

        except Exception as e:
            self.log(f"YouTube search error: {e}")
            return []
