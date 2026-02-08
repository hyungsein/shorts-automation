"""
🎬 Shorts Automation - Main Entry Point
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import settings
from .models import ContentType
from .workflows import ShortsWorkflow

app = typer.Typer(
    name="shorts-automation",
    help="🎬 AI-powered YouTube Shorts automation",
)
console = Console()


@app.command()
def generate(
    count: int = typer.Option(
        1,
        "--count",
        "-c",
        help="Number of shorts to generate",
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-cat",
        help="Category: 인간관계, 직장생활, 연애, 심리, 공감, 레전드썰, 꿀팁, 충격",
    ),
    topic: str = typer.Option(
        None,
        "--topic",
        "-t",
        help="Direct topic input (e.g. '이런 친구는 손절해라')",
    ),
    search: str = typer.Option(
        None,
        "--search",
        "-s",
        help="YouTube search query for reference",
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--fast",
        help="Strict supervisor mode (default: fast)",
    ),
):
    """Generate YouTube Shorts automatically - just run it!"""

    mode_text = "👨‍💼 STRICT" if strict else "🚀 FAST"

    # 주제 소스 결정
    if topic:
        source = f"직접입력: {topic[:20]}..."
        content_type = ContentType.CUSTOM
    elif search:
        source = f"YouTube 검색: {search}"
        content_type = ContentType.YOUTUBE_SEARCH
    else:
        source = f"자동생성 ({category or '랜덤 카테고리'})"
        content_type = ContentType.AUTO

    console.print(
        Panel.fit(
            "[bold blue]🎬 Shorts Automation[/bold blue]\n"
            f"📌 {source}\n"
            f"🔢 {count}개 생성\n"
            f"⚡ {mode_text} 모드",
            title="Starting",
        ))

    # Run workflow
    workflow = ShortsWorkflow(strict_mode=strict)

    async def run_batch():
        results = []
        for i in range(count):
            console.print(
                f"\n[cyan]━━━ Generating short {i+1}/{count} ━━━[/cyan]")
            result = await workflow.run(
                content_type=content_type,
                category=category,
                topic=topic,
                search_query=search,
            )
            results.append(result)
        return results

    results = asyncio.run(run_batch())

    # Summary
    successful = [r for r in results if r is not None]
    console.print("\n")
    console.print(
        Panel.fit(
            f"[green]✅ Generated {len(successful)}/{len(results)} shorts![/green]\n"
            f"Output directory: {settings.output_dir}",
            title="Complete",
        ))


@app.command()
def config():
    """Show current configuration"""
    console.print(
        Panel.fit(
            f"[bold]⚙️ Configuration[/bold]\n\n"
            f"Output Dir: {settings.output_dir}\n"
            f"Language: {settings.default_language}\n"
            f"TTS Voice: {settings.tts.default_voice}\n\n"
            f"[dim]API Keys configured:[/dim]\n"
            f"  AWS Bedrock: {'✅' if settings.aws.access_key_id else '⚡ (CLI)'}\n"
            f"  TypeCast: {'✅' if settings.tts.typecast_api_key else '❌'}\n"
            f"  YouTube: {'✅' if settings.youtube.api_key else '❌'}\n"
            f"  Stable Diffusion: {settings.sd.api_url}",
            title="Settings",
        ))


@app.command()
def init():
    """Initialize project (create .env file)"""
    env_path = Path(".env")
    example_path = Path(".env.example")

    if env_path.exists():
        console.print("[yellow].env file already exists![/yellow]")
        return

    if example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        console.print("[green]✅ Created .env file from .env.example[/green]")
        console.print("[dim]Please edit .env and add your API keys[/dim]")
    else:
        console.print("[red].env.example not found![/red]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
