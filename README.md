# 🎬 Shorts Automation

AI-powered YouTube Shorts automation using **LangGraph** + **MCP**

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Trend    │───▶│ Script   │───▶│ Voice    │          │
│  │ Agent    │    │ Agent    │    │ Agent    │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       │              │               │                  │
│       ▼              ▼               ▼                  │
│   Reddit API     Claude API     ElevenLabs             │
│                                                         │
│  ┌──────────┐    ┌──────────┐                          │
│  │ Video    │───▶│ Upload   │                          │
│  │ Agent    │    │ Agent    │                          │
│  └──────────┘    └──────────┘                          │
│       │              │                                  │
│       ▼              ▼                                  │
│   MoviePy        YouTube API                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
python -m src.main generate --type reddit_story --count 3
```

## 📁 Project Structure

```
shorts-automation/
├── src/
│   ├── agents/
│   │   ├── trend_agent.py    # 🔥 Trend discovery
│   │   ├── script_agent.py   # 📝 Script generation
│   │   ├── voice_agent.py    # 🎙️ TTS generation
│   │   ├── video_agent.py    # 🎬 Video creation
│   │   └── upload_agent.py   # 📤 YouTube upload
│   ├── workflows/
│   │   └── main_workflow.py  # 🔄 LangGraph workflow
│   ├── config.py             # ⚙️ Configuration
│   ├── models.py             # 📦 Data models
│   └── main.py               # 🎯 CLI entry point
├── output/                    # Generated videos
├── .env.example              # Environment template
└── pyproject.toml            # Dependencies
```

## 🎯 Content Types

| Type | Description | Source |
|------|-------------|--------|
| `reddit_story` | Viral Reddit stories | r/AmItheAsshole, r/tifu |
| `scary_story` | Horror stories | r/nosleep, r/creepypasta |
| `fun_facts` | Interesting facts | r/todayilearned |
| `motivation` | Motivational content | r/GetMotivated |

## 🔑 Required API Keys

| Service | Purpose | Get it at |
|---------|---------|-----------|
| AWS Bedrock | Script generation (Claude) | [AWS Console](https://console.aws.amazon.com/bedrock) |
| ElevenLabs | Text-to-Speech | [elevenlabs.io](https://elevenlabs.io) |
| Reddit | Content scraping | [reddit.com/prefs/apps](https://reddit.com/prefs/apps) |
| YouTube | Video upload | [console.cloud.google.com](https://console.cloud.google.com) |
| Pexels | Stock videos | [pexels.com/api](https://pexels.com/api) |

## 📊 Revenue Projection

| Month | Videos | Subscribers | Est. Revenue |
|-------|--------|-------------|--------------|
| 1 | 90 | 2K | ₩0 |
| 3 | 270 | 20K | ₩30-50만 |
| 6 | 540 | 100K | ₩100-200만 |
| 12 | 1000+ | 300K+ | ₩300-500만 |

## ⚠️ Disclaimer

- Follow YouTube's Community Guidelines
- Use royalty-free background videos and music
- Review generated content before uploading
- Respect copyright and platform policies

## 📝 License

MIT License - Use at your own risk!

---

Made with ❤️ and AI
