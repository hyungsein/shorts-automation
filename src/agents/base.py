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
        # AWS CLI credentials (~/.aws/credentials) 자동 사용
        # .env에 명시하면 그걸 우선 사용
        client_kwargs = {"region_name": settings.aws.region}

        # .env에 키가 있으면 명시적으로 사용
        if settings.aws.access_key_id and settings.aws.secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws.access_key_id
            client_kwargs[
                "aws_secret_access_key"] = settings.aws.secret_access_key

        bedrock_client = boto3.client("bedrock-runtime", **client_kwargs)

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
