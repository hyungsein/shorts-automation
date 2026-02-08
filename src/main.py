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
    content_type: str = typer.Option(
        "reddit_story",
        "--type",
        "-t",
        help="Content type: reddit_story, scary_story, fun_facts, motivation",
    ),
    count: int = typer.Option(
        1,
        "--count",
        "-c",
        help="Number of shorts to generate",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        "-s/-ns",
        help="Enable strict supervisor mode (default: on)",
    ),
):
    """Generate YouTube Shorts automatically"""

    mode_text = "👨‍💼 STRICT MODE" if strict else "🚀 FAST MODE"

    console.print(
        Panel.fit(
            "[bold blue]🎬 Shorts Automation[/bold blue]\n"
            f"Content Type: {content_type}\n"
            f"Count: {count}\n"
            f"Mode: {mode_text}",
            title="Starting",
        ))

    # Parse content type
    try:
        ct = ContentType(content_type)
    except ValueError:
        console.print(f"[red]Invalid content type: {content_type}[/red]")
        console.print(f"Available: {[e.value for e in ContentType]}")
        raise typer.Exit(1)

    # Run workflow
    workflow = ShortsWorkflow(strict_mode=strict)

    async def run_batch():
        results = []
        for i in range(count):
            console.print(f"\n[cyan]Generating short {i+1}/{count}...[/cyan]")
            result = await workflow.run(content_type=ct)
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
            f"  Reddit: {'✅' if settings.reddit.client_id else '❌'}\n"
            f"  Stable Diffusion: {'✅' if settings.sd.api_url else '❌'}",
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
