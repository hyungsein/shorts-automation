"""
🔥 Trend Agent - Discovers trending content for shorts
"""

import asyncio
from typing import Optional

import praw
from praw.models import Submission

from ..config import settings
from ..models import ContentType, TrendData
from .base import BaseAgent


class TrendAgent(BaseAgent[list[TrendData]]):
    """Agent for discovering trending content"""
    
    @property
    def name(self) -> str:
        return "🔥 TrendAgent"
    
    def __init__(self):
        super().__init__()
        self.reddit: Optional[praw.Reddit] = None
        self._init_reddit()
    
    def _init_reddit(self) -> None:
        """Initialize Reddit client"""
        if settings.reddit.client_id:
            self.reddit = praw.Reddit(
                client_id=settings.reddit.client_id,
                client_secret=settings.reddit.client_secret,
                user_agent=settings.reddit.user_agent,
            )
    
    async def run(
        self,
        content_type: ContentType = ContentType.REDDIT_STORY,
        limit: int = 10,
    ) -> list[TrendData]:
        """Fetch trending content based on type"""
        self.log(f"Fetching {content_type.value} trends...")
        
        if content_type == ContentType.REDDIT_STORY:
            return await self._fetch_reddit_stories(limit)
        elif content_type == ContentType.SCARY_STORY:
            return await self._fetch_scary_stories(limit)
        elif content_type == ContentType.FUN_FACTS:
            return await self._fetch_fun_facts(limit)
        elif content_type == ContentType.MOTIVATION:
            return await self._fetch_motivation(limit)
        else:
            self.log(f"Unknown content type: {content_type}")
            return []
    
    async def _fetch_reddit_stories(self, limit: int) -> list[TrendData]:
        """Fetch trending stories from Reddit"""
        if not self.reddit:
            self.log("Reddit client not initialized")
            return []
        
        trends = []
        subreddits = ["AmItheAsshole", "tifu", "MaliciousCompliance", "pettyrevenge"]
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    if self._is_valid_submission(submission):
                        trends.append(self._submission_to_trend(submission))
            except Exception as e:
                self.log(f"Error fetching from r/{subreddit_name}: {e}")
        
        self.log(f"Found {len(trends)} trending stories")
        return trends
    
    async def _fetch_scary_stories(self, limit: int) -> list[TrendData]:
        """Fetch scary stories from Reddit"""
        if not self.reddit:
            return []
        
        trends = []
        subreddits = ["nosleep", "creepypasta", "shortscarystories", "TwoSentenceHorror"]
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    if self._is_valid_submission(submission):
                        trend = self._submission_to_trend(submission)
                        trend.content_type = ContentType.SCARY_STORY
                        trends.append(trend)
            except Exception as e:
                self.log(f"Error fetching from r/{subreddit_name}: {e}")
        
        return trends
    
    async def _fetch_fun_facts(self, limit: int) -> list[TrendData]:
        """Fetch fun facts from Reddit"""
        if not self.reddit:
            return []
        
        trends = []
        subreddits = ["todayilearned", "interestingasfuck", "Damnthatsinteresting"]
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    trend = TrendData(
                        title=submission.title,
                        source=f"r/{subreddit_name}",
                        url=submission.url,
                        score=submission.score,
                        content=submission.title,  # TIL posts have content in title
                        content_type=ContentType.FUN_FACTS,
                    )
                    trends.append(trend)
            except Exception as e:
                self.log(f"Error fetching from r/{subreddit_name}: {e}")
        
        return trends
    
    async def _fetch_motivation(self, limit: int) -> list[TrendData]:
        """Fetch motivational content"""
        if not self.reddit:
            return []
        
        trends = []
        subreddits = ["GetMotivated", "quotes", "LifeProTips"]
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    trend = TrendData(
                        title=submission.title,
                        source=f"r/{subreddit_name}",
                        url=submission.url,
                        score=submission.score,
                        content=submission.selftext or submission.title,
                        content_type=ContentType.MOTIVATION,
                    )
                    trends.append(trend)
            except Exception as e:
                self.log(f"Error fetching from r/{subreddit_name}: {e}")
        
        return trends
    
    def _is_valid_submission(self, submission: Submission) -> bool:
        """Check if submission is valid for shorts"""
        # Skip if no text content
        if not submission.selftext:
            return False
        
        # Skip if too short or too long
        word_count = len(submission.selftext.split())
        if word_count < 100 or word_count > 2000:
            return False
        
        # Skip NSFW content
        if submission.over_18:
            return False
        
        return True
    
    def _submission_to_trend(self, submission: Submission) -> TrendData:
        """Convert Reddit submission to TrendData"""
        return TrendData(
            title=submission.title,
            source=f"r/{submission.subreddit.display_name}",
            url=f"https://reddit.com{submission.permalink}",
            score=submission.score,
            content=submission.selftext,
            content_type=ContentType.REDDIT_STORY,
        )
