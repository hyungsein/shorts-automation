"""
🤖 Base Agent Class
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

import boto3
from langchain_aws import ChatBedrock
from pydantic import BaseModel

from ..config import settings

T = TypeVar("T")


class BaseAgent(ABC, Generic[T]):
    """Base class for all agents"""
    
    def __init__(self):
        # Initialize Bedrock client
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws.region,
            aws_access_key_id=settings.aws.access_key_id,
            aws_secret_access_key=settings.aws.secret_access_key,
        )
        
        self.llm = ChatBedrock(
            model_id=settings.aws.model_id,
            client=bedrock_client,
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.7,
            },
        )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name"""
        pass
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> T:
        """Execute the agent's main task"""
        pass
    
    def log(self, message: str) -> None:
        """Log a message"""
        print(f"[{self.name}] {message}")
