# 🎬 Shorts Automation

AI-powered YouTube Shorts automation with **LangGraph** + **Supervisor Agent**

```
┌─────────────────────────────────────────────────────────────┐
│                    👨‍💼 SUPERVISOR                           │
│            (각 단계마다 품질 검토 & OK 사인)                  │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────┬───────────┬───┴───────┬───────────┬─────────┐
    ▼         ▼           ▼           ▼           ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐
│ Trend │→│Script │→│  Image  │→│  Voice  │→│ FINAL   │→│ Video │
│ Agent │ │ Agent │ │  Agent  │ │  Agent  │ │ REVIEW  │ │ Agent │
└───────┘ └───────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘
    │         │           │           │
    ▼         ▼           ▼           ▼
 Reddit    Claude    Stable       TypeCast
  API      Sonnet    Diffusion    (한국어 TTS)
            4.5      (Local)
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Stable Diffusion (Local)

```bash
# Automatic1111 or Forge 실행 필요
# http://127.0.0.1:7860
```

### 4. Generate Shorts

```bash
# 감독 모드 ON (깐깐하게 품질 체크)
python -m src.main generate --type reddit_story

# 감독 모드 OFF (빠르게)
python -m src.main generate --type reddit_story --no-strict

# 여러 개 생성
python -m src.main generate --type scary_story --count 3
```

## 📁 Project Structure

```
shorts-automation/
├── src/
│   ├── agents/
│   │   ├── base.py            # 🤖 Base agent (Bedrock)
│   │   ├── supervisor_agent.py # 👨‍💼 깐깐한 감독
│   │   ├── trend_agent.py     # 🔥 Reddit 트렌드 수집
│   │   ├── script_agent.py    # 📝 스크립트 생성
│   │   ├── image_agent.py     # 🎨 Stable Diffusion 이미지
│   │   ├── voice_agent.py     # 🎙️ ElevenLabs TTS
│   │   └── video_agent.py     # 🎬 영상 합성
│   ├── workflows/
│   │   └── main_workflow.py   # 🔄 LangGraph 워크플로우
│   ├── config.py              # ⚙️ 설정
│   ├── models.py              # 📦 데이터 모델
│   └── main.py                # 🎯 CLI
├── output/                    # 생성된 영상
├── .env.example               # 환경변수 템플릿
└── requirements.txt           # 의존성
```

## 🎯 Content Types

| Type | Description | Source |
|------|-------------|--------|
| `reddit_story` | 바이럴 레딧 스토리 | r/AmItheAsshole, r/tifu |
| `scary_story` | 공포 이야기 | r/nosleep, r/creepypasta |
| `fun_facts` | 흥미로운 사실 | r/todayilearned |
| `motivation` | 동기부여 콘텐츠 | r/GetMotivated |

## 🔑 Required API Keys

| Service | Purpose | Get it at |
|---------|---------|-----------|
| AWS Bedrock | Claude Sonnet 4.5 | [AWS Console](https://console.aws.amazon.com/bedrock) |
| TypeCast | TTS (한국 쇼츠 "그 목소리") | [biz.typecast.ai](https://biz.typecast.ai) |
| Reddit | 콘텐츠 수집 | [reddit.com/prefs/apps](https://reddit.com/prefs/apps) |

## 👨‍💼 Supervisor Mode

감독 Agent가 각 단계 결과물을 평가:

| 점수 | 결과 | 설명 |
|------|------|------|
| 9-10 | ✅ APPROVED | 바이럴 각! |
| 7-8 | 🔄 | 수정하면 좋겠지만 통과 |
| 5-6 | 🔄 NEEDS_REVISION | 재시도 |
| 1-4 | ❌ REJECTED | 다시해 |

```bash
# 감독 ON (기본값) - 품질 보장
python -m src.main generate -t reddit_story

# 감독 OFF - 빠르게 테스트
python -m src.main generate -t reddit_story --no-strict
```

## 💰 예상 비용 (하루 3개 × 30일)

| Service | Monthly Cost |
|---------|-------------|
| AWS Bedrock (Claude) | ~$10 |
| TypeCast (Starter) | $9 (2시간/월) |
| Stable Diffusion | $0 (로컬) |
| **Total** | **~$19/월** |

## ⚠️ Disclaimer

This tool is for educational purposes. Always follow platform guidelines and respect copyright.
