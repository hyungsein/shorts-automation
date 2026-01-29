"""
📝 Script Agent - Generates viral scripts for shorts
"""

from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..models import ContentType, Script, TrendData
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

Output format:
HOOK:
[Your hook here]

BODY:
[Your main content here]

CTA:
[Your call to action here]
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
            "language": language,
            "title": trend.title,
            "source": trend.source,
            "content": trend.content[:3000],  # Limit content length
            "content_type": trend.content_type.value,
        })
        
        # Parse response
        script = self._parse_script(response.content)
        script.duration_estimate = target_duration
        
        self.log(f"Script generated: {len(script.full_text)} chars")
        return script
    
    def _parse_script(self, response: str) -> Script:
        """Parse LLM response into Script object"""
        hook = ""
        body = ""
        cta = ""
        
        current_section = None
        lines = response.split("\n")
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if line_upper.startswith("HOOK:"):
                current_section = "hook"
                # Get content after "HOOK:" if on same line
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
            elif current_section and line.strip():
                if current_section == "hook":
                    hook += " " + line.strip() if hook else line.strip()
                elif current_section == "body":
                    body += " " + line.strip() if body else line.strip()
                elif current_section == "cta":
                    cta += " " + line.strip() if cta else line.strip()
        
        script = Script(
            hook=hook.strip(),
            body=body.strip(),
            cta=cta.strip(),
        )
        script.combine()
        
        return script
    
    async def generate_metadata(
        self,
        script: Script,
        trend: TrendData,
    ) -> dict:
        """Generate YouTube metadata (title, description, tags)"""
        self.log("Generating metadata...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a YouTube SEO expert. Generate metadata that maximizes views.
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
