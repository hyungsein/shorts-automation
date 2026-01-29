"""
📤 Upload Agent - Uploads videos to YouTube
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from ..config import settings
from ..models import ShortVideo, VideoResult
from .base import BaseAgent


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class UploadAgent(BaseAgent[str]):
    """Agent for uploading videos to YouTube"""
    
    @property
    def name(self) -> str:
        return "📤 UploadAgent"
    
    def __init__(self):
        super().__init__()
        self.youtube = None
        self.credentials: Optional[Credentials] = None
    
    async def authenticate(self, credentials_file: Path) -> bool:
        """Authenticate with YouTube API"""
        self.log("Authenticating with YouTube...")
        
        try:
            # Check for existing token
            token_path = credentials_file.parent / "token.json"
            
            if token_path.exists():
                self.credentials = Credentials.from_authorized_user_file(
                    str(token_path), SCOPES
                )
            
            # If no valid credentials, run OAuth flow
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    from google.auth.transport.requests import Request
                    self.credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_file), SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)
                
                # Save token for future use
                with open(token_path, "w") as f:
                    f.write(self.credentials.to_json())
            
            # Build YouTube service
            self.youtube = build("youtube", "v3", credentials=self.credentials)
            self.log("Authentication successful!")
            return True
            
        except Exception as e:
            self.log(f"Authentication failed: {e}")
            return False
    
    async def run(
        self,
        video: VideoResult,
        title: str,
        description: str,
        tags: list[str],
        schedule_time: Optional[datetime] = None,
    ) -> str:
        """Upload video to YouTube"""
        if not self.youtube:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        self.log(f"Uploading: {title}")
        
        # Prepare video metadata
        body = {
            "snippet": {
                "title": title[:100],  # YouTube title limit
                "description": description[:5000],
                "tags": tags[:500],  # Tag limit
                "categoryId": "22",  # People & Blogs
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }
        
        # Add schedule if specified
        if schedule_time:
            body["status"]["privacyStatus"] = "private"
            body["status"]["publishAt"] = schedule_time.isoformat() + "Z"
        
        # Create media upload
        media = MediaFileUpload(
            str(video.file_path),
            mimetype="video/mp4",
            resumable=True,
        )
        
        # Execute upload
        try:
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    self.log(f"Upload progress: {int(status.progress() * 100)}%")
            
            video_id = response["id"]
            video_url = f"https://youtube.com/shorts/{video_id}"
            
            self.log(f"Upload complete! {video_url}")
            return video_id
            
        except Exception as e:
            self.log(f"Upload failed: {e}")
            raise
    
    async def get_optimal_upload_time(self) -> datetime:
        """Calculate optimal upload time based on audience analytics"""
        # Default: Schedule for peak hours (6 PM - 9 PM)
        now = datetime.now()
        
        # If before 6 PM today, schedule for today 6 PM
        if now.hour < 18:
            return now.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # Otherwise schedule for tomorrow 6 PM
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=18, minute=0, second=0, microsecond=0)
    
    async def get_channel_stats(self) -> dict:
        """Get channel statistics"""
        if not self.youtube:
            return {}
        
        try:
            request = self.youtube.channels().list(
                part="statistics,snippet",
                mine=True,
            )
            response = request.execute()
            
            if response["items"]:
                channel = response["items"][0]
                return {
                    "title": channel["snippet"]["title"],
                    "subscribers": channel["statistics"]["subscriberCount"],
                    "views": channel["statistics"]["viewCount"],
                    "videos": channel["statistics"]["videoCount"],
                }
            return {}
            
        except Exception as e:
            self.log(f"Error getting channel stats: {e}")
            return {}
